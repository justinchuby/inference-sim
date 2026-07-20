import type {
  SpeculativeStateGroupConfig,
  SpeculativeStateLifetime,
  SpeculativeStateRole,
} from "./speculative.js";

export const SPECULATIVE_FAMILY_CONTRACT_REVISION = 1;

export type SpeculativeProposerFamily =
  | "prompt_lookup"
  | "draft_model"
  | "mtp"
  | "eagle3"
  | "shared_kv"
  | "self_speculative";

export type SpeculativeFamilySupport = "onnx_genai_current" | "design_only";

export type SpeculativeProposerExecution =
  | "cpu_lookup"
  | "separate_model"
  | "sidecar"
  | "shared_kv_assistant"
  | "target_early_exit";

export interface SpeculativeEligibility {
  readonly proposerAvailable: boolean;
  readonly decoding: "greedy" | "sampling";
  readonly grammarActive: boolean;
  readonly targetKvAvailable: boolean;
  readonly targetHiddenOutputCount: number;
  readonly sharedKvGroupCount: number;
  readonly targetLayerCount?: number;
  readonly earlyExitLayer?: number;
  readonly allowDesignOnly?: boolean;
}

export interface SpeculativeFamilyContract {
  readonly revision: typeof SPECULATIVE_FAMILY_CONTRACT_REVISION;
  readonly family: SpeculativeProposerFamily;
  readonly support: SpeculativeFamilySupport;
  readonly execution: SpeculativeProposerExecution;
  readonly targetHiddenOutputCount: number;
  readonly requiresSharedKvGroups: boolean;
  readonly proposerCostScale: number;
  readonly state: readonly {
    readonly role: SpeculativeStateRole;
    readonly owner: "target" | "proposer";
    readonly lifetime: SpeculativeStateLifetime;
  }[];
}

const TARGET_STATE = {
  role: "target_kv" as const,
  owner: "target" as const,
  lifetime: "committed_prefix" as const,
};

const CONTRACTS: Readonly<Record<
  SpeculativeProposerFamily,
  SpeculativeFamilyContract
>> = {
  prompt_lookup: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "prompt_lookup",
    support: "onnx_genai_current",
    execution: "cpu_lookup",
    targetHiddenOutputCount: 0,
    requiresSharedKvGroups: false,
    proposerCostScale: 0.08,
    state: [TARGET_STATE],
  },
  draft_model: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "draft_model",
    support: "onnx_genai_current",
    execution: "separate_model",
    targetHiddenOutputCount: 0,
    requiresSharedKvGroups: false,
    proposerCostScale: 1,
    state: [
      TARGET_STATE,
      {
        role: "draft_kv",
        owner: "proposer",
        lifetime: "committed_prefix",
      },
    ],
  },
  mtp: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "mtp",
    support: "onnx_genai_current",
    execution: "sidecar",
    targetHiddenOutputCount: 1,
    requiresSharedKvGroups: false,
    proposerCostScale: 0.32,
    state: [
      TARGET_STATE,
      {
        role: "sidecar_kv",
        owner: "proposer",
        lifetime: "proposal_local",
      },
      {
        role: "recurrent_state",
        owner: "proposer",
        lifetime: "proposal_local",
      },
    ],
  },
  eagle3: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "eagle3",
    support: "onnx_genai_current",
    execution: "sidecar",
    targetHiddenOutputCount: 3,
    requiresSharedKvGroups: false,
    proposerCostScale: 0.46,
    state: [
      TARGET_STATE,
      {
        role: "sidecar_kv",
        owner: "proposer",
        lifetime: "proposal_local",
      },
      {
        role: "recurrent_state",
        owner: "proposer",
        lifetime: "proposal_local",
      },
    ],
  },
  shared_kv: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "shared_kv",
    support: "onnx_genai_current",
    execution: "shared_kv_assistant",
    targetHiddenOutputCount: 1,
    requiresSharedKvGroups: true,
    proposerCostScale: 0.38,
    state: [
      TARGET_STATE,
      {
        role: "shared_kv_lease",
        owner: "target",
        lifetime: "borrowed",
      },
      {
        role: "recurrent_state",
        owner: "proposer",
        lifetime: "proposal_local",
      },
    ],
  },
  self_speculative: {
    revision: SPECULATIVE_FAMILY_CONTRACT_REVISION,
    family: "self_speculative",
    support: "design_only",
    execution: "target_early_exit",
    targetHiddenOutputCount: 0,
    requiresSharedKvGroups: false,
    proposerCostScale: 0.55,
    state: [
      TARGET_STATE,
      {
        role: "early_exit_state",
        owner: "proposer",
        lifetime: "proposal_local",
      },
    ],
  },
};

export class SpeculativeEligibilityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SpeculativeEligibilityError";
  }
}

export function speculativeFamilyContract(
  family: SpeculativeProposerFamily,
): SpeculativeFamilyContract {
  return CONTRACTS[family];
}

export function validateSpeculativeEligibility(
  family: SpeculativeProposerFamily,
  eligibility: SpeculativeEligibility,
): SpeculativeFamilyContract {
  const contract = speculativeFamilyContract(family);
  assertNonNegativeSafeInteger(
    eligibility.targetHiddenOutputCount,
    "targetHiddenOutputCount",
  );
  assertNonNegativeSafeInteger(
    eligibility.sharedKvGroupCount,
    "sharedKvGroupCount",
  );
  if (!eligibility.proposerAvailable) {
    throw new SpeculativeEligibilityError(
      `${family} proposer is not available`,
    );
  }
  if (!eligibility.targetKvAvailable) {
    throw new SpeculativeEligibilityError(
      `${family} requires target KV support`,
    );
  }
  if (eligibility.grammarActive) {
    throw new SpeculativeEligibilityError(
      `${family} is incompatible with grammar constraints in the current onnx-genai loop`,
    );
  }
  if (eligibility.decoding !== "greedy") {
    throw new SpeculativeEligibilityError(
      `${family} requires greedy or temperature-zero decoding`,
    );
  }
  if (
    eligibility.targetHiddenOutputCount
    < contract.targetHiddenOutputCount
  ) {
    throw new SpeculativeEligibilityError(
      `${family} requires ${contract.targetHiddenOutputCount} target hidden output(s)`,
    );
  }
  if (
    contract.requiresSharedKvGroups
    && eligibility.sharedKvGroupCount < 1
  ) {
    throw new SpeculativeEligibilityError(
      `${family} requires at least one declared shared-KV group`,
    );
  }
  if (contract.support === "design_only" && !eligibility.allowDesignOnly) {
    throw new SpeculativeEligibilityError(
      `${family} is design-only and requires allowDesignOnly`,
    );
  }
  if (family === "self_speculative") {
    const targetLayers = eligibility.targetLayerCount;
    const earlyExit = eligibility.earlyExitLayer;
    if (
      targetLayers === undefined
      || earlyExit === undefined
      || !Number.isSafeInteger(targetLayers)
      || !Number.isSafeInteger(earlyExit)
      || targetLayers < 2
      || earlyExit < 1
      || earlyExit >= targetLayers
    ) {
      throw new SpeculativeEligibilityError(
        "self_speculative requires 1 <= earlyExitLayer < targetLayerCount",
      );
    }
  }
  return contract;
}

export function validateSpeculativeStateGroups(
  family: SpeculativeProposerFamily,
  groups: readonly SpeculativeStateGroupConfig[],
): void {
  const contract = speculativeFamilyContract(family);
  const normalized = groups.map((group) => ({
    role: group.role ?? (
      group.owner === "target" ? "target_kv" : "draft_kv"
    ),
    owner: group.owner,
    lifetime: group.lifetime ?? "committed_prefix",
  }));
  const allowedRoles = new Set<SpeculativeStateRole>([
    ...contract.state.map((state) => state.role),
    "target_aux",
  ]);
  for (const group of normalized) {
    if (
      group.role === "target_aux"
      && (
        group.owner !== "target"
        || group.lifetime !== "committed_prefix"
      )
    ) {
      throw new SpeculativeEligibilityError(
        "target_aux state must be target-owned with committed_prefix lifetime",
      );
    }
    if (!allowedRoles.has(group.role)) {
      throw new SpeculativeEligibilityError(
        `${family} does not permit state role ${group.role}`,
      );
    }
  }
  for (const required of contract.state) {
    if (!normalized.some((group) => (
      group.role === required.role
      && group.owner === required.owner
      && group.lifetime === required.lifetime
    ))) {
      throw new SpeculativeEligibilityError(
        `${family} requires ${required.owner} ${required.role} with ${required.lifetime} lifetime`,
      );
    }
  }
}

export function defaultSpeculativeEligibility(
  family: SpeculativeProposerFamily,
): SpeculativeEligibility {
  const contract = speculativeFamilyContract(family);
  return {
    proposerAvailable: true,
    decoding: "greedy",
    grammarActive: false,
    targetKvAvailable: true,
    targetHiddenOutputCount: contract.targetHiddenOutputCount,
    sharedKvGroupCount: contract.requiresSharedKvGroups ? 1 : 0,
    ...(family === "self_speculative"
      ? {
          targetLayerCount: 32,
          earlyExitLayer: 16,
          allowDesignOnly: true,
        }
      : {}),
  };
}

export function buildSpeculativeStateGroups(
  family: SpeculativeProposerFamily,
  capacityTokens: number,
  maxRollbackTokens: number,
): readonly SpeculativeStateGroupConfig[] {
  const contract = speculativeFamilyContract(family);
  return contract.state.map((state, index) => ({
    id: `${family}:${state.role}:${index}`,
    owner: state.owner,
    role: state.role,
    lifetime: state.lifetime,
    capacityTokens: state.lifetime === "proposal_local"
      ? Math.max(1, maxRollbackTokens)
      : capacityTokens,
    rollbackProtection: state.lifetime === "proposal_local"
      ? {
          kind: "bounded_snapshot" as const,
          maxRollbackTokens,
        }
      : { kind: "non_destructive_tail" as const },
  }));
}

function assertNonNegativeSafeInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new SpeculativeEligibilityError(
      `${label} must be a non-negative safe integer; got ${value}`,
    );
  }
}
