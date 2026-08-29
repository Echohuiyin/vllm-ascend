# Famem Allocator

Famem is an optional Ascend A2/A3 allocator for sleep-mode model weights and KV cache. A native per-device HBM server
preallocates and owns one resident physical-memory pool; each vLLM Worker and its Copier import and map that pool in
their own processes. This page is the feature's specification, design, deployment, and validation guide.

The feature switches select one allocator path:

| `multiproc_pipe` | `use_fast_map_allocator` | Path |
| --- | --- | --- |
| Off | Off | Default Camem four-field handles |
| On | Off | Camem Copier control pipeline + per-allocation Camem backend |
| On | On | The same Copier control pipeline + server-arena Famem backend |

Famem without `multiproc_pipe` is rejected. Both Copier paths use an immutable initialization image and reject physical
Level 2 sleep.

## Specification

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** define required behavior.

| Area | Requirement |
| --- | --- |
| Enablement | Famem MUST be explicitly enabled by `additional_config.use_fast_map_allocator`. |
| Executor | It MUST use the `mp` executor, `spawn`, V1 multiprocessing, sleep mode, and eager execution. |
| Weight format | `VLLM_ASCEND_ENABLE_NZ` MUST be `0`; FRACTAL_NZ cannot be restored by this pipeline. |
| Server | One compatible native HBM server MUST preallocate its pool before publishing a socket for each physical NPU. |
| Capacity | Server `--size-gib` MUST be 1–4096 GiB per NPU; Worker `size_gib` MUST match it exactly. |
| Mapping | A virtual mapping MUST NOT outlive the importing process's physical handle. |
| Ownership | Only the HBM server owns original allocation handles; Worker and Copier own imported handles. |
| Release | Every importer MUST unmap and call `aclrtFreePhysical` on its local imported handle. |
| Weights | Weights MUST remain immutable after the Copier captures its image. |
| Sleep | Level 0 MAY pause scheduling only; every physical sleep MUST use Level 1. |
| Resume | A layer MUST NOT execute before its restore event succeeds. |
| Failure | An ambiguous or partial lifecycle transition MUST fail the Worker group closed. |
| Cleanup | Original handles MUST remain resident until server shutdown; importer metadata remains until cleanup succeeds. |

Famem v1 supports:

- CANN 8.5.1 and torch-npu 2.9.0 on Ascend A2/A3, vLLM V1, source builds with
  `COMPILE_CUSTOM_KERNELS=1`, and `enforce_eager=True`.
- `Qwen2ForCausalLM`, `DeepseekV2ForCausalLM`, and `DeepseekV3ForCausalLM`.
- At most one active Worker/Copier lease per NPU; multiple sleeping model processes may remain connected. TP 1, 2, 4,
  or 8; no PP, DP, CP, Ray, or Xlite.
- Static weights. LoRA, speculative decoding, dynamic EPLB, EPLB map recording, `layer_sharding`,
  `calculate_kv_scales`, Worker extensions, callable `collective_rpc`, and runtime weight APIs such as `apply_model` are
  rejected. A static `expert_map_path` may run before the backup is captured.
- Level 1 `sleep`, which retains the initialization-time weight backup and discards KV cache. Level 2 is rejected because
  its runtime weight reload conflicts with the Copier's immutable image.

Graph execution and `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` are unsupported. Workspaces outside the
weight/KV memory-pool scopes continue to use torch-npu's default allocator.

## Architecture and ownership

```text
frontend -> EngineCore -> Executor -> Worker -> CaMemAllocator-compatible lifecycle
                                             |
                                             +-> CopierProcess -> one Copier control loop
                                                               |-> Camem memory backend
                                                               `-> Famem memory backend

Famem Worker ---- UDS lease ----> native HBM server ----> resident original physical pool
      |                                  ^
      `---- imported mapping             `---- sole owner of original handles
Famem Copier ---- imported mapping
```

Camem is the compatibility contract, not a second implementation to bypass. Both allocator modes use the same Worker
entry points, `CopierProcess`, control messages, response/health checks, and layer-ready events. Famem supplies only the
physical-memory operations required by that loop: initial import/map and D2H backup, importer-only unmap on suspend,
same-handle reimport/remap and layer-wise H2D on resume, and importer cleanup on close. It does not define another Copier
process or lifecycle state machine.

The Worker creates the original descriptor, recovery, and control queues and installs them through
`CaMemAllocator.set_desc_queue`, `set_npu_recover_queue`, and `set_ctrl_queue`; allocator startup consumes those same
objects, and the static control queue remains the original command queue. The spawned process actually targets the
six-argument `copier_main` with the Worker's real tensor-parallel size and local rank, not a parallel Famem entry point.
A process-argument adapter carries that command queue together with the response queue and backend selector solely for
bounded acknowledgements and health reporting; it does not create a second command protocol or lifecycle loop.

### Camem compatibility contract

The complete call chain keeps the following signatures. Famem MUST remain substitutable at the Worker and Copier
boundaries; backend-specific descriptors MUST NOT change the frontend, engine, executor, or Worker RPC contract.

| Layer | Required interface |
| --- | --- |
| Offline/API | `LLM.suspend(level=1, mode="abort")`; `LLM.resume(tags=None)`; HTTP `/suspend?level=&mode=` and repeated `/resume?tags=` |
| Engine | `LLMEngine.suspend(level=1, mode="abort")`; `resume(tags=None)`; matching `AsyncLLM` methods |
| Core client | Sync and async `suspend(level=1, mode="abort")` and `resume(tags=None)` utility calls |
| EngineCore | `suspend(level=1, mode="abort")`: pause with `clear_cache=level >= 1`, then executor suspend; `resume(tags=None)`: executor resume, then scheduler resume |
| Executor | `suspend(level=1)` and `resume(tags=None)` with the same sleeping-state guards as `sleep`/`wake_up` |
| NPUWorker | `sleep(level=1)`, `suspend(level=1)`, `wake_up(tags=None)`, `resume(tags=None)`, and the common shared-pool model-load path |
| CaMemAllocator | `sleep(offload_tags=None)`, `suspend(offload_tags=None)`, `wake_up(tags=None)`, `resume(tags=None)`, descriptor send/wait, aligned-address lookup, four-/five-tuple callbacks, and local/shared pool contexts |
| Copier | `copier_main(desc_queue, npu_recover_queue, ctrl_queue, tp_size, local_rank, layer_ready_events)`, `should_preload`, `parse_layer_index`, and `_preload_weight` |
| Native Camem | Four-tuple local and five-tuple share allocation APIs; create/map, unmap/free, export/import, Copier import/free, and share-handle import bindings |

The native compatibility ABI is exact: `init_module_share(python_malloc_share, python_free_share)` enables the original
five-tuple callbacks; `python_create_and_map_share(device, size, d_mem, p_mem_handle, old_shareable_handle)` ignores the
expired export token, recreates backing at the retained virtual reservation, and returns one new shareable-handle
integer. `python_share_memHandle_import` has a matching `python_share_memHandle_free`; every successful import therefore
has an explicit `aclrtFreePhysical` release path.

Layer names retain the original sentinels: `unknown=-2`, `pub=-1`, `layers.N=N`, and invalid names `=-3`. `unknown`,
`pub`, and every numbered layer are present even when empty, so every live weight allocation belongs to exactly one
immutable backup group. Public and unknown data are restored before layer zero; a numbered layer event is published only
after that layer's H2D copy succeeds.

The native server is the only server implementation and the authority for lease, activation epoch, physical layout, and
original-handle ownership. It exposes protocol v5 over a trusted Unix-domain socket: a 24-byte network-order header,
payloads up to 4096 bytes, and `HELLO`, `ACQUIRE`, `SLEEP`, `WAKE`, and `RELEASE` operations. The server validates frame
metadata, peer credentials, connection-bound sessions, activation epoch, and the one-active-lease rule before changing
state.
Do not expose its socket directory to untrusted users.

Each connection follows this state machine:

| State | Lease access to resident pool | Transition |
| --- | --- | --- |
| Connected | None | `ACQUIRE` authorizes import → Active |
| Active | Authorized; Worker mapped; Copier may remain mapped after restore until suspend | `SLEEP` → Sleeping; `RELEASE` → Released |
| Sleeping | No importer mapping or local handle | `WAKE` reauthorizes import → Active; `RELEASE` → Released |
| Released | None | Terminal; the server pool remains resident |

`HELLO` may finish while another model is active. Conflicting `ACQUIRE` or `WAKE` returns `FamemBusyError` without
consuming an epoch or poisoning either lease. Resume is retryable only when every rank reports Busy; a mixed
success/Busy result fails closed because rank state has diverged.

At startup the server builds one fixed arena with at most two extents: available 1 GiB pages followed by a 2 MiB-page
remainder, or one complete 2 MiB-page extent when the preferred layout is unavailable. It allocates and exports every
extent before binding the socket; any allocation/export failure fails startup and publishes no socket. `ACQUIRE` and
`WAKE` only authorize the existing shareable handles and return that fixed layout—they never allocate or export physical
memory. The Worker maps the extents into one stable contiguous virtual range. A 512-byte-aligned bump allocator places
weights first and KV cache second; individual frees update diagnostics but do not reuse bump space.

### CANN handle lifetime

`aclrtMallocPhysical` creates each server original physical handle. The server exports it once during startup, producing
a shareable token. `aclrtMemImportFromShareableHandle` creates a different, process-local handle in each consumer. An
importer must first call `aclrtUnmapMem`, then `aclrtFreePhysical(imported_handle)`.

Despite its name, the Copier's `aclrtFreePhysical` call destroys only its local imported handle/reference. The Copier
never calls Famem's server-free operation and never owns the original handle. A successful importer barrier makes the
pool eligible for another lease but does not release HBM; only server shutdown frees original handles.

The inspected CANN driver corroborates why the barrier is required: `devmm_ioctl_mem_import_local_server` increments the
shared block's `occupied_cnt`, mapping separately increments a process-local `map_ref`, and imported-handle release
decrements shared occupancy. Thus an imported but unmapped handle may still keep physical memory occupied. This is
implementation evidence, not a replacement for the public CANN contract that every importer must release its handle.

CANN's internal reference count cannot replace Famem server accounting. It cannot decide which model owns the active
lease, reject stale epochs, coordinate the Worker/Copier barrier, or recover a disconnected client. The mandatory
ownership order is:

```text
Copier unmap + free local imported handles
  -> Worker unmap + free local imported handles
  -> server publish lease as sleeping/released and pool as available
  -> next activation reauthorize + import + map the same resident pool
```

On a normal `SLEEP` or `RELEASE`, the server trusts the authenticated Worker to complete this importer barrier; it
cannot query CANN's importer count. Driver reference counting protects the underlying allocation lifetime but does not
provide the server's lease or epoch state.

`aclrtMemSetPidToShareableHandle` adds authorized PIDs to an exported token; it does not provide a documented revocation
operation. Consequently, PIDs authorized by an earlier activation remain able to import a token they retained. Famem v1
therefore assumes a cooperative trusted domain: the server, all model processes, and the socket directory belong to the
same trusted operator, and sleeping/released clients follow the protocol rather than re-importing out of turn. The UDS
lease prevents accidental concurrent activation, but it is not a security boundary against a malicious former lease.
Do not share one server pool across mutually untrusted tenants.
The inspected CANN drivers cap retained PID-whitelist entries at 65,535, so deployments with many long-lived sleeping
or released processes must account for that implementation limit.

## Deployment

Run one persistent server per physical NPU. The launcher replaces itself with the installed
`vllm_ascend_hbm_server` executable. Server and Workers must use compatible CANN installations, the same UID, and the
same socket-directory path. UUID-named sockets identify physical devices even when local device ordinals differ.

For containerized Workers, run the server in the host PID namespace with host `/proc`, and bind-mount the socket
directory at the same path. Worker containers may use a private PID namespace: the host server observes their host PIDs,
which match the CANN bare TGIDs. `HELLO` rejects a Worker PID that differs from the socket peer or a Copier that is not
its live direct child. Start the servers before vLLM and give each server the full per-rank resident-pool budget; for
TP=2 with 48 GiB pools:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_FAMEM_SOCKET_DIR=/run/vllm-ascend/famem
install -d -m 0750 "$VLLM_ASCEND_FAMEM_SOCKET_DIR"
vllm-ascend-hbm-server --device 0 --size-gib 48 &
vllm-ascend-hbm-server --device 1 --size-gib 48 &
```

Each TP rank has an independent server pool, lease, and Worker/Copier pair. Shareable handles are authorized for the
Worker and Copier bare TGIDs with `aclrtMemSetPidToShareableHandle`; handles are never shared across ranks. Repeated
activation authorizes the fixed exported handles again and assigns a server-global, monotonically increasing epoch.

Launch vLLM with the same integer arena size per NPU. A mismatch is rejected before any import or mapping:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export VLLM_ASCEND_ENABLE_NZ=0

vllm serve /models/Qwen2.5-7B-Instruct \
  --tensor-parallel-size 2 \
  --distributed-executor-backend mp \
  --enable-sleep-mode --enforce-eager --multiproc-pipe \
  --use-famem-allocator --use-famem-allocator-size 48
```

For programmatic construction, configure `spawn` before importing vLLM and protect the entrypoint:

```python
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
os.environ.setdefault("VLLM_ASCEND_ENABLE_NZ", "0")

from vllm import LLM


if __name__ == "__main__":
    llm = LLM(
        model="/models/Qwen2.5-7B-Instruct",
        enable_sleep_mode=True,
        enforce_eager=True,
        distributed_executor_backend="mp",
        additional_config={
            "multiproc_pipe": True,
            "use_fast_map_allocator": {"enabled": True, "size_gib": 48},
        },
    )
```

Only `enabled` and `size_gib` are valid Famem configuration fields. The former top-level `famem_allocator` key is
rejected with a rename error. `VLLM_ASCEND_MULTIPROC_PIPE_TIMEOUT` defaults to 600 seconds and bounds Copier waits and
the complete lifecycle RPC, including enqueue and all Worker replies. Timeout makes the Worker group fail closed.

## Lifecycle and Copier pipeline

During model loading, the Worker allocates in Famem and publishes tensor spans only after post-processing has produced
the final inference layout. The Copier imports the activation's shareable handles, maps them at a process-local stable
address, captures the final weight image in host memory, and acknowledges readiness. Discovery covers Parameters,
registered buffers, and persistent backend tensors exposed through hooks, including SFA, ALLTOALL routing, FlashComm2,
and RoPE state. Scratch workspace that is overwritten before use is excluded.

Because the Copier image already contains the post-processed inference layout, pipeline `wake_up`/`resume` restores its
bytes without running the ordinary Camem MoE transpose a second time. The legacy non-pipeline wake path retains that
layout repair for weight reloads.

The two Level 1 APIs retain their original Camem distinction. Camem `sleep/wake_up` uses the Worker's temporary D2H/H2D
backup; the Copier drops its imported handles on sleep and remains quiescent after wake. Camem `suspend/resume` uses the
Copier's immutable initialization image and layer events. Famem has no Worker-owned physical pages or per-allocation
backup, so both Famem pairs release the lease and restore through that same Copier image.

Use only matched lifecycle pairs:

- `sleep(level=0)` only pauses scheduling and leaves allocator state untouched; `wake_up(tags=["scheduling"])` resumes
  scheduling without mapping memory.
- `sleep(level=1)` retains all required weight backups and releases importer mappings and handles. Camem releases its
  Worker-owned physical memory; Famem releases the lease while the server-owned physical pool remains resident.
  `wake_up(tags=None)` remaps the participants and restores the matching Worker or Copier image.
- `sleep(level=2)` and partial weight wake are rejected before scheduler, mapping, or lease state changes.
- `suspend` uses the Copier's initialization image and releases the current activation without freeing the server pool.
- `resume` authorizes and imports the same resident pool under a new epoch, remaps Worker and Copier, restores global
  weights and layer zero before admitting requests, then overlaps later-layer restore with computation. Each layer waits
  for its own success event.

`suspend` is a release barrier:

1. Engine admission stops; outstanding work drains, aborts, or is frozen and requeued according to the pause mode.
2. Copier finishes queued copies, unmaps, releases all local imported handles, clears events, and acknowledges.
3. Worker synchronizes the device, unmaps, and releases all local imported handles.
4. Worker sends `SLEEP`; the server marks the lease sleeping and the resident pool available. It retains the original
   handles, exported tokens, layout, and epoch authority.

A suspend immediately after resume waits for final-layer restore; it does not require a dummy inference. Lifecycle
operations are serialized, and `is_sleeping()` changes only after the physical transition completes.

The supported multi-instance handoff is:

```text
model 1 start -> model 1 suspend
model 2 start -> model 2 suspend
model 1 resume
```

A sleeping lease has no local imports or mapping, but the fixed server-owned pool continues to consume its configured
HBM budget. All models using that server must request exactly the pool capacity. Each successful `ACQUIRE` or `WAKE`
returns the same shareable handles and a new global epoch while preserving each process's reserved virtual base.
Operators must wait for successful `suspend` before starting or waking the next model. The next model overwrites pool
contents; a resumed model reconstructs its weights from its retained host backup before inference.

## Failure and cleanup

Lifecycle transitions are not generally idempotent: one TP rank or the server may commit before the caller observes an
error. Only a Busy response received before state change is retryable. Mapping, copying, event, protocol, timeout, or
partial-release ambiguity triggers local cleanup and then `WorkerFatalError`; the executor terminates all peer Workers.
Restart the model process. If the HBM server must restart, first stop every active or sleeping Famem model on that
physical NPU and confirm its importers exited, then restart the server and reconstruct the models; leases and epochs
are not persisted or reconnected across a server restart. Stopping the server releases the resident HBM pool.

The frontend→EngineCore→Worker→Copier chain arms Linux `PR_SET_PDEATHSIG(SIGKILL)` before NPU initialization and checks
the expected parent PID after arming, closing the parent-exit race. Copier creation is restricted to the Worker's main
thread because Linux associates the death signal with the creating thread. On an active client disconnect, the server
waits for both registered Worker and Copier PIDs before making the pool available to another lease; it uses pidfds where
available and process start time as the PID-reuse fallback. The disconnect does not free original handles.

Additional rules:

- Native allocation OOM does not advance `heap_top`; an OOM inside a weight/KV pool transaction is terminal.
- Failed unmap/free retains metadata for cleanup retry and poisons uncertain state against reuse.
- Startup allocation rollback and shutdown retry outstanding original handles before finalization; startup failure never
  publishes a socket.
- Running the HBM server in a private or sibling PID namespace, or without host `/proc`, is unsupported because it cannot
  authenticate and monitor CANN bare TGIDs.

## Capacity and diagnostics

The server's `--size-gib` is a resident per-NPU reservation made before any model connects, not a per-lease or aggregate
budget. The Worker's `use_fast_map_allocator.size_gib` is an exact compatibility assertion for that pool. The bump
high-water mark includes temporary tensors created by transpose, repack, or quantization post-processing. Freed blocks
are not reused, so the arena may need to exceed the final weight footprint.

KV capacity is calibrated from actual K/V, MLA, DSA, Mamba, hybrid, or shared tensor requests plus 2 MiB address padding,
torch-npu compatibility padding, and caching-segment rounding. All ranks use the largest common block count that fits;
a second preflight runs before entering the KV pool. Do not apply `gpu_memory_utilization` to `size_gib` again.
`PYTORCH_NPU_ALLOC_CONF=roundup_power2_divisions:...` is unsupported because it changes calibrated request sizes.

`get_famem_stats` reports Worker-side weight/bump counters, NPU usage, and the imported page layout. `heap_top` must not
exceed capacity; actual KV bytes must not exceed the planned upper bound; extent byte counters must sum to capacity while
active. Worker state `SLEEPING` means its
local mapping and imported handles are gone—not that the server returned HBM. Observable free HBM should drop once when
the server starts, remain approximately flat across acquire/suspend/resume/model handoff, and return only after server
shutdown; allow for unrelated runtime allocations when measuring on hardware.

## Implementation and validation

| Responsibility | Location |
| --- | --- |
| Configuration and platform checks | `vllm_ascend/device_allocator/famem_config.py`, `vllm_ascend/platform.py` |
| Client, allocator, and memory-pool callbacks | `vllm_ascend/device_allocator/famem.py` |
| Common Worker/Copier control pipeline | `vllm_ascend/worker/worker.py`, `vllm_ascend/worker/copier.py` |
| Famem memory-backend adapter | `vllm_ascend/worker/famem_copier.py` |
| Native allocator, protocol, and server | `csrc/famem_allocator.cpp`, `csrc/famem_hbm_server.c` |
| Focused unit tests | `tests/ut/device_allocator/` |
| Consolidated hardware E2E | `tests/e2e/singlecard/test_famem.py` |

Run the focused CPU/fake-ACL suite:

```bash
PYTHONPATH=../vllm:. .venv/bin/python -m pytest -q tests/ut/device_allocator
```

Build with `COMPILE_CUSTOM_KERNELS=1`; confirm the wheel contains `libvllm_ascend_famem.so` and
`vllm_ascend_hbm_server`, then inspect the ABI and dependencies:

```bash
nm -D vllm_ascend/libvllm_ascend_famem.so | grep -E \
  'famem_server_allocate_export|famem_server_authorize|famem_worker_get_allocations'
ldd vllm_ascend/libvllm_ascend_famem.so
ldd vllm_ascend/vllm_ascend_hbm_server
```

The single Famem E2E case covers synchronous sleep/wake and a two-checkpoint suspend handoff. The checkpoints configured
by `FAMEM_E2E_MODEL` and `FAMEM_E2E_MODEL_2` must produce different baselines so the test proves weight restoration.

```bash
pytest -sv tests/e2e/singlecard/test_famem.py
```

For target qualification, repeat the case at each supported TP size and verify Worker/server termination cleanup on an
idle NPU. Record model and dependency revisions, device, arena size, allocator statistics, and pre/post-test HBM.
