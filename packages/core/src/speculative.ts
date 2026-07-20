export type SpeculativeStateOwner = "target" | "proposer";

export type SpeculativeStateRole =
  | "target_kv"
  | "target_aux"
  | "draft_kv"
  | "sidecar_kv"
  | "recurrent_state"
  | "shared_kv_lease"
  | "early_exit_state";

export type SpeculativeStateLifetime =
  | "committed_prefix"
  | "proposal_local"
  | "borrowed";

export type RollbackProtection =
  | { readonly kind: "non_destructive_tail" }
  | {
      readonly kind: "bounded_snapshot";
      readonly maxRollbackTokens: number;
    };

export interface SpeculativeStateGroupConfig {
  readonly id: string;
  readonly owner: SpeculativeStateOwner;
  readonly role?: SpeculativeStateRole;
  readonly lifetime?: SpeculativeStateLifetime;
  readonly capacityTokens: number;
  readonly rollbackProtection: RollbackProtection;
}

export interface SpeculativeStateGroupSnapshot {
  readonly id: string;
  readonly owner: SpeculativeStateOwner;
  readonly role: SpeculativeStateRole;
  readonly lifetime: SpeculativeStateLifetime;
  readonly logicalLength: number;
  readonly highWaterLength: number;
  readonly capacityTokens: number;
}

export interface SpeculativeIterationInput {
  readonly draftTokenCount: number;
  readonly acceptedDraftTokenCount: number;
}

export interface SpeculativeIterationResult {
  readonly iterationId: number;
  readonly checkpointId: number;
  readonly baseTokenLength: number;
  readonly draftTokenCount: number;
  readonly acceptedDraftTokenCount: number;
  readonly rejectedDraftTokenCount: number;
  readonly targetAuthoritativeTokenCount: 1;
  readonly committedTokenCount: number;
  readonly finalTokenLength: number;
  readonly stateGroups: readonly SpeculativeStateGroupSnapshot[];
}

interface MutableStateGroup {
  id: string;
  owner: SpeculativeStateOwner;
  role: SpeculativeStateRole;
  lifetime: SpeculativeStateLifetime;
  logicalLength: number;
  highWaterLength: number;
  capacityTokens: number;
  rollbackProtection: RollbackProtection;
}

interface CompositeCheckpoint {
  id: number;
  generation: number;
  baseTokenLength: number;
  lengths: ReadonlyMap<string, number>;
}

export class SpeculativeProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeProtocolError";
  }
}

/**
 * Models the transactional state semantics of one linear speculative decoder.
 *
 * Token values and acceptance policy live outside this class. Each iteration
 * verifies all drafts on the target, restores every state group to the accepted
 * prefix, and commits exactly one target-authoritative correction/bonus token.
 */
export class SpeculativeTransactionSimulator {
  private readonly groups = new Map<string, MutableStateGroup>();
  private currentTokenLength: number;
  private generation = 0;
  private nextCheckpointId = 1;
  private nextIterationId = 1;

  constructor(
    initialTokenLength: number,
    stateGroups: readonly SpeculativeStateGroupConfig[],
  ) {
    assertNonNegativeSafeInteger(initialTokenLength, "initialTokenLength");
    if (stateGroups.length === 0) {
      throw new SpeculativeProtocolError(
        "at least one speculative state group is required",
      );
    }
    if (!stateGroups.some((group) => group.owner === "target")) {
      throw new SpeculativeProtocolError("at least one target state group is required");
    }

    for (const config of stateGroups) {
      if (config.id.length === 0 || this.groups.has(config.id)) {
        throw new SpeculativeProtocolError(
          `state group id must be non-empty and unique; got ${config.id}`,
        );
      }
      assertNonNegativeSafeInteger(config.capacityTokens, `${config.id} capacity`);
      const lifetime = config.lifetime ?? "committed_prefix";
      if (
        lifetime === "committed_prefix"
        && config.capacityTokens < initialTokenLength
      ) {
        throw new SpeculativeProtocolError(
          `${config.id} capacity ${config.capacityTokens} is below initial length ${initialTokenLength}`,
        );
      }
      validateRollbackProtection(config.id, config.rollbackProtection);
      this.groups.set(config.id, {
        id: config.id,
        owner: config.owner,
        role: config.role ?? (
          config.owner === "target" ? "target_kv" : "draft_kv"
        ),
        lifetime,
        logicalLength: lifetime === "committed_prefix"
          ? initialTokenLength
          : 0,
        highWaterLength: lifetime === "committed_prefix"
          ? initialTokenLength
          : 0,
        capacityTokens: config.capacityTokens,
        rollbackProtection: config.rollbackProtection,
      });
    }
    this.currentTokenLength = initialTokenLength;
    this.assertInvariants();
  }

  get tokenLength(): number {
    return this.currentTokenLength;
  }

  snapshot(): readonly SpeculativeStateGroupSnapshot[] {
    return [...this.groups.values()]
      .map(({ rollbackProtection: _rollbackProtection, ...group }) => ({ ...group }))
      .sort((left, right) => left.id.localeCompare(right.id));
  }

  runIteration(input: SpeculativeIterationInput): SpeculativeIterationResult {
    assertNonNegativeSafeInteger(input.draftTokenCount, "draftTokenCount");
    assertNonNegativeSafeInteger(
      input.acceptedDraftTokenCount,
      "acceptedDraftTokenCount",
    );
    if (input.acceptedDraftTokenCount > input.draftTokenCount) {
      throw new SpeculativeProtocolError(
        `accepted ${input.acceptedDraftTokenCount} drafts but only ${input.draftTokenCount} were proposed`,
      );
    }

    const checkpoint = this.checkpoint();
    const candidateLength = checkedAdd(
      checkpoint.baseTokenLength,
      input.draftTokenCount,
      "candidate length",
    );
    const finalTokenLength = checkedAdd(
      checkpoint.baseTokenLength,
      checkedAdd(input.acceptedDraftTokenCount, 1, "committed tokens"),
      "final token length",
    );

    this.preflight(
      input.draftTokenCount,
      candidateLength,
      finalTokenLength,
    );

    for (const group of this.groups.values()) {
      const candidateGroupLength = group.lifetime === "proposal_local"
        ? input.draftTokenCount
        : group.lifetime === "borrowed"
          ? checkpoint.baseTokenLength
          : candidateLength;
      group.logicalLength = candidateGroupLength;
      group.highWaterLength = Math.max(
        group.highWaterLength,
        candidateGroupLength,
      );
    }

    // Restore is anchored to this checkpoint, never to a naked token length.
    const acceptedPrefixLength = checkedAdd(
      checkpoint.baseTokenLength,
      input.acceptedDraftTokenCount,
      "accepted prefix length",
    );
    this.restorePrefix(checkpoint, acceptedPrefixLength);

    // The correction token on rejection, or bonus token on full acceptance, is
    // target-authoritative. It is a new state write after restore, not a cursor
    // increase that resurrects a stale speculative slot.
    for (const group of this.groups.values()) {
      group.logicalLength = group.lifetime === "committed_prefix"
        ? finalTokenLength
        : 0;
      group.highWaterLength = Math.max(
        group.highWaterLength,
        group.logicalLength,
      );
    }
    this.currentTokenLength = finalTokenLength;
    this.generation++;
    this.assertInvariants();

    return {
      iterationId: this.nextIterationId++,
      checkpointId: checkpoint.id,
      baseTokenLength: checkpoint.baseTokenLength,
      draftTokenCount: input.draftTokenCount,
      acceptedDraftTokenCount: input.acceptedDraftTokenCount,
      rejectedDraftTokenCount:
        input.draftTokenCount - input.acceptedDraftTokenCount,
      targetAuthoritativeTokenCount: 1,
      committedTokenCount: input.acceptedDraftTokenCount + 1,
      finalTokenLength,
      stateGroups: this.snapshot(),
    };
  }

  private checkpoint(): CompositeCheckpoint {
    return {
      id: this.nextCheckpointId++,
      generation: this.generation,
      baseTokenLength: this.currentTokenLength,
      lengths: new Map(
        [...this.groups.values()].map((group) => [group.id, group.logicalLength]),
      ),
    };
  }

  private preflight(
    rollbackHorizon: number,
    candidateLength: number,
    finalTokenLength: number,
  ): void {
    for (const group of this.groups.values()) {
      const requiredCapacity = group.lifetime === "proposal_local"
        ? rollbackHorizon
        : Math.max(candidateLength, finalTokenLength);
      if (requiredCapacity > group.capacityTokens) {
        throw new SpeculativeProtocolError(
          `${group.id} capacity ${group.capacityTokens} cannot hold candidate/final length`,
        );
      }
      if (
        group.lifetime !== "borrowed"
        &&
        group.rollbackProtection.kind === "bounded_snapshot"
        && rollbackHorizon > group.rollbackProtection.maxRollbackTokens
      ) {
        throw new SpeculativeProtocolError(
          `${group.id} rollback horizon ${rollbackHorizon} exceeds snapshot bound ${group.rollbackProtection.maxRollbackTokens}`,
        );
      }
    }
  }

  private restorePrefix(
    checkpoint: CompositeCheckpoint,
    acceptedPrefixLength: number,
  ): void {
    if (checkpoint.generation !== this.generation) {
      throw new SpeculativeProtocolError(
        `checkpoint ${checkpoint.id} belongs to stale generation ${checkpoint.generation}`,
      );
    }
    if (
      acceptedPrefixLength < checkpoint.baseTokenLength
      || acceptedPrefixLength > this.currentCandidateHighWater()
    ) {
      throw new SpeculativeProtocolError(
        `restore prefix ${acceptedPrefixLength} is outside checkpoint candidate range`,
      );
    }
    for (const group of this.groups.values()) {
      const checkpointLength = checkpoint.lengths.get(group.id);
      const expectedCheckpointLength = group.lifetime === "committed_prefix"
        ? checkpoint.baseTokenLength
        : 0;
      if (
        checkpointLength === undefined
        || checkpointLength !== expectedCheckpointLength
      ) {
        throw new SpeculativeProtocolError(
          `checkpoint ${checkpoint.id} has inconsistent state for ${group.id}`,
        );
      }
      if (
        group.lifetime === "committed_prefix"
        && acceptedPrefixLength > group.logicalLength
      ) {
        throw new SpeculativeProtocolError(
          `restore prefix ${acceptedPrefixLength} exceeds ${group.id} candidate length ${group.logicalLength}`,
        );
      }
      group.logicalLength = group.lifetime === "proposal_local"
        ? 0
        : group.lifetime === "borrowed"
          ? 0
          : acceptedPrefixLength;
    }
  }

  private currentCandidateHighWater(): number {
    return Math.max(...[...this.groups.values()].map((group) => group.logicalLength));
  }

  private assertInvariants(): void {
    for (const group of this.groups.values()) {
      const expectedLength = group.lifetime === "committed_prefix"
        ? this.currentTokenLength
        : 0;
      if (group.logicalLength !== expectedLength) {
        throw new SpeculativeProtocolError(
          `${group.id} length ${group.logicalLength} diverges from committed length ${this.currentTokenLength}`,
        );
      }
      if (
        group.logicalLength > group.highWaterLength
        || group.highWaterLength > group.capacityTokens
      ) {
        throw new SpeculativeProtocolError(
          `${group.id} violates logical/high-water/capacity ordering`,
        );
      }
    }
  }
}

function validateRollbackProtection(
  groupId: string,
  protection: RollbackProtection,
): void {
  if (protection.kind === "bounded_snapshot") {
    assertNonNegativeSafeInteger(
      protection.maxRollbackTokens,
      `${groupId} maxRollbackTokens`,
    );
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    throw new SpeculativeProtocolError(
      `${label} exceeds Number.MAX_SAFE_INTEGER`,
    );
  }
  return result;
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeProtocolError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}
