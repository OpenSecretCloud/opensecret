# Maple pairing authority integrity

Maple pairing authority is stored in host-visible PostgreSQL. A MAC on each
row detects modification, but it cannot by itself detect a row that was
deleted, an append-only high-water table whose newest generation was removed,
or a complete child graph deleted in foreign-key order. Maple therefore uses
cascading authenticated heads to commit to the complete authority inventory.

This is an integrity and fail-closed design. It does not make PostgreSQL
available or private, and its rollback guarantee ends at the authenticated
root described below.

## Authenticated hierarchy

Every head has a versioned, domain-separated canonical encoding, a monotonic
authority revision, explicit counts and inventory digest, and an enclave-keyed
MAC. The hierarchy is mandatory after feature activation:

1. An **account head**, scoped to one `(users.uuid, users.project_id)` parent,
   commits to the complete Maple authority inventory for that account. The
   inventory covers Maple devices, device-registration operations, retired
   registration-operation tombstones, pairing lineages, pairings, pairing
   operations, host revocation states, revocation high-water generations,
   revocation events, reset-clear obligations, and reset-clear admission
   leaves. It records explicit category counts, including both the
   number of high-water installation groups and the total generations.
2. A **project head**, anchored to one `org_projects` parent, authenticates the
   parent's exact immutable `(id, org_id, uuid, client_id)` identity tuple and
   commits to the complete ordered set of its account heads. The public Maple
   wire subject is `client_id`; it is read only from the locked, MAC-verified
   project head rather than from a separate route lookup.
3. An **organization head**, anchored to one `orgs` parent, commits to the
   complete ordered set of its project heads.
4. One mandatory **global head** commits to the complete ordered set of
   organization heads and to the complete lifetime issuer-key fingerprint
   registry. Its issuer-key count and inventory digest are covered by the
   global-head MAC, so the registry is part of the same authenticated root as
   the tenant hierarchy.

Canonical inventories use fixed domain and category tags, length-delimited
fields, explicit counts, and stable ordering by durable scoped identifiers.
Every included row or child head is authenticated and structurally validated
before its canonical contribution is accepted. A verifier always derives the
applicable complete inventory from the database and compares the resulting
counts and digest with the stored head; it does not trust cached or
incrementally maintained counts as proof of completeness.

Inventory recomputation streams deterministically ordered, bounded database
pages into the canonical digest instead of loading the whole scope into
memory. Hard per-category counts and quotas are checked before allocating or
processing attacker-controlled collections. Exceeding a bound fails closed.
MAC and digest comparisons use constant-time equality.

High-water rows, reset-clear obligations, retired installation lineages, and
retired-operation tombstones retain keyed pseudonymous lookup and
authority-scope digests. Those digests let the enclave enumerate the complete
history belonging to an account without storing raw account, project,
installation, operation, or device identifiers in those retained parents. A
private reset-clear admission leaf deliberately keeps its pair UUID and
incarnation so the enclave can prove exact canonical completeness; the leaf's
authorization digest and MAC remain confidential and the leaf is never returned
on the public wire.

## V1 capacity envelope

Each account may retain at most 1,024 high-water installation groups and 4,096
high-water generations across those groups. It may also retain at most 4,096
reset-clear obligations and 524,288 reset-clear admission leaves, with at most
128 admission leaves committed by any one obligation. The installation-group
limit is equivalent to 32 complete fleets of 32 installations. After reserving
one generation for every possible retained group, the generation limit leaves
3,072 generations, enough for at least 96 additional complete 32-installation
fleet resets after generation-one allocation.

Live device-registration operations and retired registration-operation
tombstones share one lifetime account cap: their combined count cannot exceed
32,768. Terminal installation-lineage retirements have a separate lifetime
cap of 1,024, equal to the maximum number of retained installation groups.
Obligation ciphertext pages contain at most 64 rows; admission, tombstone, and
retirement metadata pages contain at most 256 rows. Every category has a
separate authenticated account-head count and hard capacity check, so no
retained history can grow outside the account-head count envelope.

The operation-record quota is exactly five records per pairing: at most one
for each v1 operation kind, enforced by a unique
`(pairing, operation_kind)` constraint. Exact replay uses the existing record;
it cannot grow an unbounded operation log.

The service may retain at most 1,024 issuer-key identities over its lifetime.
This is a permanent registry capacity, not a count of currently signing or
currently referenced keys. Issuer identities are never removed, compacted, or
reused, even after their final durable artifact or tenant has been deleted.

A registration that would create installation group 1,025, a reset that would
create total generation or obligation 4,097, admission leaf 129 for one
obligation, or account-wide admission leaf 524,289, or an operation that would
exceed the shared 32,768 registration-history bound returns a typed capacity
error and rolls back its entire transaction. Bounded, non-sensitive telemetry
reports near-capacity scopes before they become unavailable. V1 does not
silently prune, compact, or discard authenticated history. Its only capacity
escape is a verified-clean final deletion followed by explicit account
recreation; that deletion uses the coordinated local-admission-clear procedure
below.

## Frozen V1 and issuer-key continuity

The deployed pairing contract accepts exactly V1. Every Maple registration,
pairing, revocation, reset-clear, checkpoint, sync, and operation payload or
receipt version accepted from a materializer or reconstructed from durable
storage must equal `1`. A positive unknown version is not forward compatible
and fails closed before publication or replay. Supporting V2 requires an
explicitly versioned model, transcript, validation path, schema, and migration;
V1 rows and receipts are never reinterpreted under newer semantics.

The configured verification keyset is likewise exactly V1, non-empty when
present, canonically ordered by key identifier, and limited to 1,024 unique
Ed25519 public keys. Startup hashes each raw public key with SHA-256 and
reconciles the complete configured set against the global
`maple_pairing_issuer_keys` registry. Each immutable registry row binds one key
identifier to exactly one algorithm and public-key fingerprint, plus its
creation time, under a domain-separated enclave MAC. The global issuer
inventory commits to the ordered semantic tuple `(key identifier, algorithm,
fingerprint)` for every row. A key identifier cannot be rebound, one public key
cannot acquire a second identifier, and rows cannot be updated, deleted, or
truncated.

Registry reconciliation requires exact full equality, not merely that the
configured set contains keys referenced by current rows. Every persisted key
must remain in every later configuration with the same fingerprint. Rotation
therefore changes `[A]` only to a retained set such as `[A, B]`; configuring
`[B]`, remapping `A`, or omitting any historical key is a configuration
conflict. Missing configured keys may be appended, but their rows, the global
issuer count and digest, the global revision, and the root MAC change in one
transaction. The 1,024-key bound is a conservative lifetime ceiling; reaching
it requires an explicit future protocol and migration rather than key removal.

Startup also authenticates every durable issuer-bearing registration, pairing,
revocation, reset-clear, retirement, tombstone, and receipt row and proves that
each referenced key identifier is present in the registry. This reference audit
detects corrupt or unknown references; it is not a key-retirement mechanism.

## Transactions and verification

Maple authority operations acquire a fixed PostgreSQL transaction-scoped
advisory lock before inspecting or changing the hierarchy. This lock is
database/cluster-wide across application replicas and is held until commit or
rollback. It applies only to the Maple authority control plane:

- Maple device and pairing reads and writes, including revocation paging and
  acknowledgement;
- account/user, project, and organization creation or deletion, because these
  operations change mandatory hierarchy membership; and
- namespace resets and authority bootstrap or audit.

Ordinary agent data traffic, such as conversation and tool-result storage,
does not take this lock.

Lock acquisition uses a bounded `pg_try_advisory_xact_lock` polling loop. The
loop may establish the transaction's serializable snapshot, but it reads no
authority data while waiting. Immediately after acquisition, the first
authority-row access locks the complete global head `FOR UPDATE`, validates
its Active shape and enclave-keyed MAC, then constant-time compares its
authenticated issuer inventory digest with the exact keyset digest pinned by
that process. This comparison occurs before the activation-marker lookup and
before any scoped authority read or mutation. A mismatch is an issuer
configuration conflict and fails closed rather than allowing a stale replica to
create or consume authority with a different verifier set. The activation
marker is verified only after that replica fence passes. Because every
cooperating mutation advances that root, a waiter whose snapshot predates the
preceding commit aborts with `40001` and returns a typed busy result before it
can read any scoped authority state. Only the explicit bootstrap path may
accept the exact Pending sentinel, after also proving the activation marker and
every authority leaf/head are absent.

CREATE and REGISTER also fingerprint the verification keyset supplied to their
materializer boundary and constant-time compare that semantic digest with the
process-pinned digest before acquiring a connection or reading scoped state.
An internal caller therefore cannot substitute a same-ID/different-key verifier
while leaving the authenticated registry and transaction fence unchanged.

Bootstrap is the sole exception to the configured-digest comparison against an
already Active root because it must authorize a monotonic registry expansion.
It first verifies the old complete tree and registry, appends only missing
configured fingerprints, advances the global root once, verifies the resulting
tree, and pins the resulting digest for ordinary transactions. After `[A, B]`
commits, an old `[A]` replica fails its next authority transaction before scoped
access. A transaction that already held the global lock may finish under the
old root, but `A` remains permanently retained, so its artifacts remain
verifiable. Operators must therefore deploy the same full retained keyset to
all replicas; adding a key deliberately creates a fail-closed cutover for any
replica that has not received the new configuration.

V1 deliberately accepts serialization of Maple device and pairing reads under
this database-wide lock. Each operation uses one `SERIALIZABLE` database
transaction, selected as the transaction's first statement, with finite lock
and statement timeouts. The advisory lock coordinates cooperating application
replicas; serializable isolation supplies the stable database snapshot and
predicate-conflict detection needed when a privileged or out-of-band writer
does not honor that lock. A serialization failure never turns into partial
success. Reads and exact-idempotent operations may be retried, while a caller
retrying a mutation must preserve the exact operation identifier and signed
request body so a prior commit replays instead of publishing a second accepted
mutation.

Pairing incarnations use a PostgreSQL sequence, whose reservations are
deliberately nontransactional. A precommit reservation may therefore leave a
gap after an abort, but sequence values are never reused and only a committed,
published CREATE establishes an incarnation. Any signed candidate produced
for an aborted attempt is discarded and must never be returned. Retrying the
same operation may commit a strictly higher incarnation; replaying an already
committed operation returns its durable receipt without reserving or signing
again.

Fresh CREATE allocates and invokes its synchronous materializer inside the
authority transaction only after credential, participant, pending-reset,
exact-replay, endpoint, shape, and quota gates have passed. The callback
context obtains its public project subject only from the locked, MAC-verified
project head. The callback output is not itself trusted evidence. Before
inserting a lineage, pairing, or operation, the database independently verifies
the ticket with the process-pinned, registry-matched issuer keyset at trusted
database time, recomputes and constant-time compares the enclave-keyed
request-nonce binding, and rebinds both the actor-signed request endpoint keys
and issuer-signed ticket participant keys to the locked device identity MACs.
The callback returns only a typed signed ticket and typed response. The
database requires exact semantic equality, then canonically serializes and
AEAD-encrypts the pairing payload and operation receipt itself; callback-owned
ciphertext hashes are never treated as proof. A materialization failure can
burn only the never-reused sequence value; it cannot leave a row or publish a
candidate.

REVOKE follows the same boundary. Its mutation carries the exact actor-signed
wire request and its callback returns the typed request ticket, pair
authorization, issuer-signed revocation, and response. Before the first event
insert, the database decrypts the current authenticated pairing payload,
requires exact ticket/authorization equality, verifies the pinned issuer chain
and actor signature, and binds every participant, project, direction, stream,
sequence, reason, and timestamp to locked state. It derives the event digest
and owns canonical AEAD for the replacement pairing payload, event payload,
and receipt.

An authority transaction performs no network request or awaited asynchronous
cryptographic operation while holding the lock. Advisory-lock wait time, lock
timeout, authority callback duration (including its time waiting for and
working under the lock), and serialization aborts are exposed through bounded,
non-sensitive structured telemetry so operators can identify saturation
without logging authority material. A caller may retry an eligible operation,
but V1 does not retry inside the database layer and does not claim to measure
the caller's retry interval. V1 uses structured tracing for these signals; it
does not require a separate metrics subsystem.

Every Maple read fully recomputes and verifies the account inventory in scope
and verifies its authenticated ancestor chain. A mutation first verifies the
current complete inventory and ancestor chain, applies all row changes,
recomputes the affected heads, advances their authority revisions, and updates
their MACs in the same transaction. Parent creation installs its mandatory
head, and verified parent deletion consumes that head and updates its
ancestors, also in the same transaction. Exact-operation replays still verify
the hierarchy before returning their stored result.

Missing heads, invalid MACs, non-canonical inventories, revision or membership
mismatches, deleted high-water tails, and count or digest mismatches are
integrity failures. Reads, mutations, reset, and destructive deletion fail
closed; the service must not silently reconstruct a head from the remaining
rows.

## Bootstrap

Initial activation is serialized by the same transaction-scoped advisory
lock. If the durable activation marker is absent, genesis heads may be created
only after proving that the complete Maple device, pairing, revocation, and
authority-head inventory and the issuer registry are empty. Existing account,
project, and organization parents then receive empty child heads. The complete
configured issuer fingerprint set is inserted, and the ancestor heads, Active
global root, and activation marker are installed atomically. A Pending root
always has a zero issuer count and digest; the first Active root commits the
exact configured registry.

If any Maple authority data exists without an activated hierarchy, activation
fails closed and requires an explicit recovery decision. Once the marker is
present, startup and control-plane access require the global head and every
head implied by the parent tables to be present, MAC-valid, and consistent
with a full recomputation. A missing head is never treated as an uninitialized
scope. The advisory lock and single activation transaction also prevent two
replicas from independently bootstrapping different roots.

The destructive down migration has a separately named disposable-test guard,
but that guard is not application authorization. Production application roles
must not have migration or arbitrary DDL privileges. In particular, setting a
custom PostgreSQL parameter cannot authorize deletion or downgrade of an
active root; the active-root DML guard remains unconditional. Migration DDL is
reserved for a distinct operator role following the forward recovery process.

## Reset-clear obligations and deletion

A password or credential reset is not final account deletion. Before removing
the live device and pairing graph, the reset transaction advances the account
authority from security epoch `N` to `N + 1`, appends a fresh per-installation
revocation namespace, and persists an authenticated reset-clear obligation for
each live installation and each unresolved pending installation chain. A
terminal acknowledged installation is deliberately excluded: its lookup,
lineage, and high-water history are retired permanently and are never rewritten
or reused. The account authority epoch starts at one when the account head is
created and advances once per reset. Epochs, revocation generations, stream
identifiers, reset identifiers, event identifiers, and registration operation
identifiers are never reused.

The durable reset-clear state has two authenticated layers:

1. One append-only pseudonymous obligation row per installation per reset binds
   the installation lookup and authority scope, host identity claim, public
   reset and event identifiers, old and target namespaces, target security
   epoch, cumulative obligation count, previous event identifier, previous
   instruction-material digest, previous chain digest, state,
   acknowledgement head, encrypted signed instruction, issuer key identifier,
   and record MAC. The newest obligation is the chain head; older obligation
   rows remain authenticated history.
2. Up to 128 admission child leaves per obligation bind the complete set of
   pair admissions that reset requires that host to remove. Each private leaf
   retains the pair UUID and incarnation plus an authorization digest; the
   obligation commits to the leaves' count and aggregate digest without
   exposing any leaf publicly.

Neither layer contains a recoverable raw account, project, installation, or
device identifier. Pair UUIDs retained only in admission children identify the
canonical admission set but are neither returned to the recovering client nor
sufficient to exercise authority.

One public reset identifier names the account-wide reset batch, while each host
obligation receives its own event identifier. Hosts retain independent chains:
the same reset may therefore append several obligations sharing one reset
identifier but carrying distinct event identifiers, installation lookups, host
claims, namespaces, admission aggregates, and acknowledgements.

Each reset includes every admission then authorized for that host. At most 128
child leaves are retained per obligation; account-wide installation and row
caps remain independently enforced. Obligation and child rows are included in
the account authority inventory, streamed in deterministic bounded pages, and
committed through the account, project, organization, and global heads. Reset
appends the new epoch, obligation, child leaves, and high-water generation
before deleting live graph rows, and all changes commit or roll back together.
A crash can therefore expose either the complete previous state or the complete
new reset-clear state, never an advanced epoch without its obligation or a
deleted graph without durable clear evidence.

### Latest-head recovery

The host does not acknowledge leaves one at a time. The server lazily signs a
single latest-head `ResetClearRequired` instruction whose recursive chain
digest and cumulative count prove the complete gap-free history through the
current reset. It carries the full host identity claim, aggregate admission-set
count and digest, public reset and event identifiers, current security epoch,
target revocation checkpoint, issuer key identifier, and encrypted instruction
material. Older instructions remain authenticated chain members, but only the
current head may clear the scope.

The same exact signed pending sync is stored with an accepted registration
operation and repeated byte-for-byte by exact registration replay and by
revocation listing. The list and acknowledgement routes remain the existing
routes; reset-clear is a typed event/sync state rather than a ninth pairing
route. Public responses never expose per-pair leaves, admission identifiers,
authority-scope or installation lookup digests, request MACs, record MACs, or
other database authentication material.

Authority-bearing database inputs and retained rows do not derive blanket
`Debug`. Authority-bearing v1 wire envelopes also use custom redacted output:
only non-secret version, status, count, epoch, and paging summaries may appear,
followed by an explicit redaction marker. Stable account, project, device,
installation, operation, event, and namespace identifiers; full host claims;
issuer identifiers and signatures; digests; and exact syncs remain omitted.
Admission leaves are wholly redacted. Request-time logs must not name reset,
event, or operation identifiers or emit issuer identifiers, ciphertexts,
digests, MACs, or exact sync payloads.

After receiving the latest instruction, the same installation must durably
remove every admission covered by the cumulative chain before signing one
scope-clear acknowledgement. The host SDK represents successful durable local
clearing as a linear, non-`Clone` proof wrapper. Preparing an acknowledgement
consumes that proof exactly once and binds it to one exact operation identifier
and request body. The proof cannot prepare a second operation. Only the
resulting prepared acknowledgement request may be cloned and retried, and every
retry must be byte-identical. Clear-ACK operation identifiers are unique within
one host-registration namespace, not account-wide. The same raw operation UUID
may therefore be used independently by two different host registrations without
making either request a replay of the other.

The signed acknowledgement request directly binds the current event digest,
target revocation namespace and sequence, expected previous sequence, host
registration identifier, and exact operation identifier. The event digest
transitively binds the full issuer-signed instruction, including the complete
host identity claim, security epoch, reset/event identifiers, recursive chain
digest, and cumulative count. The server stores the exact request MAC with the
outcome. Its transaction atomically marks the latest obligation acknowledged,
links every ancestor obligation to that acknowledged head, records the exact
operation/request/receipt outcome, and advances the host checkpoint. It creates
the receipt only after locking and validating the current high-water state and
applying that transition inside the same serializable transaction, so a
concurrently committed event is reflected in the receipt's issued and
acknowledged sequences.
An acknowledgement for an ancestor, stale epoch, stale chain, incomplete host
claim, different installation, or different request body cannot clear a newer
obligation. Exact replay returns the stored receipt; a crash yields either no
acknowledgement or the entire acknowledged chain. An acknowledgement response
is always a durable operation receipt, not an authoritative statement of
current readiness. It was truthful at its original commit, but an exact replay
may be historical after later events. Only a freshly fetched and verified
registration or revocation-list sync may install `Ready` in the client.

The server implements that namespace with two separate domain-separated keyed
pseudonyms. It first derives an account-scoped host-registration lookup digest
from the raw host registration identifier, then derives the ACK-operation lookup
digest from the authority scope, that host pseudonym, and the raw operation
identifier. The current terminal obligation MAC binds the raw operation
identifier, host-registration pseudonym, request MAC, and exact receipt. The
immutable retirement row binds the same host pseudonym and nested operation
digest to the final obligation and receipt without retaining the raw host or
operation identifier. Replay requires both public identifiers, recomputes both
pseudonyms, and matches the authenticated obligation and retirement. Reusing an
operation identifier under another host registration therefore cannot retrieve,
conflict with, or discharge the first host's obligation.

If a host misses several resets, the newest instruction proves their complete
chain and one current-head acknowledgement clears them together. Every
successor retains the byte-identical full host claim from its pending
predecessor and enforces strict endpoint continuity; a changed endpoint cannot
substitute a different installation into the chain. Multiple installations
remain independent: every host chain head must be acknowledged before the
account is clean. A reset racing registration or acknowledgement is serialized
by the authority transaction; the loser observes either the prior complete
epoch or the next complete obligation and cannot bypass it.

### Registration and readiness

Every device registration request, stored operation, exact receipt, signed
checkpoint, and reset-clear sync explicitly binds the account authority
security epoch. The device row is covered by the complete account inventory at
that epoch. A client must send its known epoch. An ahead epoch, missing or
tampered retained epoch, mismatched identity claim, or reused operation
identifier fails closed. Pending reset-clear state dominates epoch equality:
even a current-epoch registration cannot become ready until the latest chain is
durably acknowledged.

The epoch is authenticated both by the account-head MAC and by the account
inventory header. A live registration operation's scope and lookup digests,
known and accepted epochs, response kind, exact encrypted sync, sync issuer and
sync digest are authenticated in both its receipt MAC and its exact inventory
leaf; category counts alone are never accepted as proof that these rows are
intact.

Registration-operation identifiers have global lifetime uniqueness across
reset. Reset therefore retains a pseudonymous, MAC-authenticated operation
tombstone binding the operation HMAC, request MAC, retired epoch, authority
scope and installation lookup digests, result kind, outcome digest, a bounded
canonical set of issuer key identifiers referenced by the response, the exact
encrypted full registration response/sync receipt version, ciphertext and
SHA-256 digest, acceptance and retirement times, and record MAC. Its encryption
AAD is derived only from durable pseudonymous facts, not a deleted device row or
raw identifier. The tombstone remains account-rooted and contains no raw
account, project, installation, device, or operation identifier. An old ready
operation cannot replay as ready after an epoch advance, and changing its body
cannot reuse its identifier.
An accepted pending registration replays the exact stored
`ResetClearRequired` sync even after acknowledgement; it never changes into a
ready result. Once acknowledged, that installation lookup is permanently
retired. A fresh registration operation using the retired installation identity
returns HTTP 409 with code `MapleInstallationRetired` and the exact message
`This Maple installation enrollment is retired; reset Remote access on this
device and enroll it again.` The three-field public error contains no IDs,
epochs, or authority material. The retired-lookup check dominates stale epoch,
materializer, and incarnation-allocation paths. An exact historical operation
or tombstone may replay only its frozen historical outcome, never current
readiness. Specifically, an accepted pre-acknowledgement registration operation
replays its byte-identical historical `ResetClearRequired` sync after ACK, and
the exact ACK operation replays its historical ACK receipt. Changing the body
while reusing either operation identifier is a conflict; a genuinely fresh
operation on the retired installation receives `MapleInstallationRetired`.
Reset converts every live registration operation to its complete tombstone in
the same transaction before deleting any device, host, or pairing row. Exact
replay decrypts the tombstone's own authenticated historical receipt without
depending on the deleted graph.
Historical replay and the retired-lineage 409 require the retained verification
keyset but not an active signing key. The service demands a signer only inside
the fresh-registration materializer after the database has passed exact replay,
retirement, epoch, identity, and recovery gates.

The current production `main` intentionally does not inject either pairing
issuer configuration into `AppStateBuilder`; this keeps every signer-dependent
pairing route unavailable while the attested signing/configuration integration
is staged. Enabling these routes requires injecting the same complete retained
verification keyset on every replica plus an active signer contained in that
keyset. This is an explicit deployment boundary, not an implicit environment
fallback.

Returning to `Ready` requires explicit enrollment with a genuinely fresh
installation-instance identifier and fresh installation identity. That new
lineage starts at the account's current epoch only after proving there is no
prior lookup history. A registration or revocation-list sync for the fresh
installation—not an acknowledgement receipt—provides the authoritative current
readiness state.

While an installation's latest obligation is pending, it may register for
recovery, list the repeated signed sync, and submit the exact clear
acknowledgement. It may not create, approve, confirm, revoke, acknowledge an
ordinary pair revocation, reserve an incarnation, replay a pairing operation,
or otherwise exercise remote authority. Issuer keys referenced by retained
instructions, acknowledgements, or registration receipts remain live under the
issuer-reference audit. Tombstone issuer-reference sets are structurally
validated and MAC-bound, then paged by issuer audit without decrypting receipt
ciphertext. Referenced checkpoint and reset-instruction keys, like every
registered issuer key, therefore remain retained permanently; final tenant
deletion does not remove them.

The clear acknowledgement atomically appends a separate pseudonymous
installation-lineage retirement row, binding the scope and lookup, host identity
MAC, retired epoch and time, final obligation/event, host-registration
pseudonym, nested ACK-operation pseudonym, exact request and receipt outcome,
and record MAC. It then removes that installation's live device, host, and
pairing rows in the same authority transaction. This durable lineage row—not an
operation tombstone—is the identity-retirement proof used by fresh registration
and authority gates. It also permanently fences reuse of the retired host
registration identifier. Exact ACK replay first resolves the retained
retirement, obligation, and receipt through the supplied host registration,
host-scoped operation identifier, and request MAC before looking for a
now-deleted live device; a changed request conflicts.
The mutation transaction repeats this exact replay check before its own live
device lookup, and the web boundary retries replay if device discovery observes
the concurrent retirement first. Thus two concurrent identical ACKs converge
on one stored receipt instead of turning the losing request into a missing-host
result.

After the clear acknowledgement, the retired installation remains barred from
ordinary pairing create/read, a fresh ACK, and every other remote-authority
operation. This is a terminal identity state, not merely a pending-reset gate.

### Final deletion

Destructive account, project, or organization deletion first verifies the
complete affected hierarchy and a clean authority predicate. Deletion is
blocked while any pairing can grant authority, any ordinary terminal
revocation remains unacknowledged, or any reset-clear chain head is pending.
There is no offline-host timeout, administrative force bypass, or inference
that an empty live graph means an old admission was cleared.

Once every host chain head is authenticated and acknowledged, final deletion
may consume the obligation and child rows, registration-operation tombstones,
installation retirements, devices and pairing graph, scoped high-water history,
and corresponding account heads. Project and organization deletion use the
same whole-subtree two-phase rule: prove every account clean before deleting
any authority row or parent. All affected ancestor heads are updated or
consumed in the same transaction. High-water, reset-clear, operation-tombstone,
and retirement history survives credential reset and disappears only through
this verified-clean final parent deletion. The global issuer-key registry is not
tenant state and is never consumed by account, project, or organization
deletion; its count and digest remain unchanged while the ordinary global
organization inventory and root revision advance.

## Threat boundary

With the enclave authentication key intact and the current global head
available, this hierarchy detects partial insertion, modification, or deletion
of Maple authority data. It also detects a coherent rollback of an account
subtree when the project head remains current, a project subtree when the
organization head remains current, and an organization subtree when the
global head remains current. Missing or inconsistent evidence causes denial
of authority operations rather than reconstruction.

A coherent rollback of the entire database, including the activation marker,
global head, issuer registry, all descendants, and their revisions and MACs, is
not detectable from this database alone. The same limitation applies to
restoring a complete older root and matching registry through snapshot or
point-in-time recovery. Detecting that event requires a monotonic checkpoint
outside the rolled-back database, such as an enclave-sealed monotonic record or
an externally witnessed checkpoint, plus an explicit reconciliation protocol.
Until that exists, this design must be
described only as protection against partial tampering and deletion, and
subtree rollback caught by a still-current authenticated ancestor.
