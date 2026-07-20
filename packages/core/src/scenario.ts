import {
  SCENARIO_SCHEMA_VERSION,
  type AllocationClass,
  type AllocationReservation,
  type MemoryDomainSpec,
  type ScenarioMemoryLedgerEntry,
  type ScenarioMemoryLedgerOptions,
  type ScenarioValidationIssue,
  type ScenarioValidationResult,
  type SimLinkSpec,
  type SimulationScenario,
  type TransferRequirement,
  type TransferRoute,
} from "./scenario-types.js";

const SCENARIO_FAMILIES = [
  "cpu_only",
  "single_discrete",
  "multi_gpu",
  "gpu_npu",
  "unified",
  "multi_node",
  "custom",
] as const;
const MEMORY_DOMAIN_KINDS = ["host", "device", "unified", "storage"] as const;
const ALLOCATION_CLASSES = [
  "pageable",
  "pinned",
  "device",
  "unified",
  "storage",
] as const;
const DEVICE_KINDS = ["cpu", "gpu", "npu"] as const;
const COMPUTE_CAPABILITIES = [
  "attention",
  "ffn",
  "collective",
  "copy",
  "sampling",
  "draft",
  "lookup",
] as const;
const LINK_KINDS = [
  "on-chip",
  "pcie",
  "nvlink",
  "ethernet",
  "infiniband",
  "thunderbolt",
  "storage",
] as const;
const ALLOCATION_PURPOSES = [
  "weights",
  "kv",
  "workspace",
  "staging",
  "cache",
  "backing",
  "checkpoint",
  "sidecar",
] as const;
const SPECULATIVE_FAMILIES = [
  "prompt_lookup",
  "draft_model",
  "mtp",
  "eagle3",
  "shared_kv",
  "self_speculative",
] as const;
const CONFIDENCE_CLASSES = [
  "exact",
  "bounded",
  "calibrated",
  "heuristic",
] as const;

export class ScenarioValidationError extends Error {
  readonly issues: readonly ScenarioValidationIssue[];

  constructor(issues: readonly ScenarioValidationIssue[]) {
    super(
      `scenario validation failed with ${issues.length} issue(s): ${
        issues.map((issue) => `${issue.path}: ${issue.message}`).join("; ")
      }`,
    );
    this.name = "ScenarioValidationError";
    this.issues = issues;
  }
}

export function assertValidScenario(scenario: SimulationScenario): void {
  const result = validateScenario(scenario);
  if (!result.valid) {
    throw new ScenarioValidationError(result.issues);
  }
}

export function validateScenario(
  scenario: SimulationScenario,
): ScenarioValidationResult {
  const issues: ScenarioValidationIssue[] = [];
  const add = (code: string, path: string, message: string): void => {
    issues.push({ code, path, message });
  };

  if (scenario.schemaVersion !== SCENARIO_SCHEMA_VERSION) {
    add(
      "schema_version",
      "schemaVersion",
      `expected ${SCENARIO_SCHEMA_VERSION}, got ${scenario.schemaVersion}`,
    );
  }
  if (scenario.id.length === 0) {
    add("empty_id", "id", "scenario id must not be empty");
  }
  validateEnum(scenario.family, SCENARIO_FAMILIES, "family", add);

  validatePositiveInteger(
    scenario.workload.batchSize,
    "workload.batchSize",
    add,
  );
  validateNonNegativeInteger(
    scenario.workload.inputSequenceLength,
    "workload.inputSequenceLength",
    add,
  );
  validateNonNegativeInteger(
    scenario.workload.outputSequenceLength,
    "workload.outputSequenceLength",
    add,
  );
  if (scenario.workload.speculative) {
    validateEnum(
      scenario.workload.speculative.family,
      SPECULATIVE_FAMILIES,
      "workload.speculative.family",
      add,
    );
    validateNonNegativeInteger(
      scenario.workload.speculative.maxAdditionalTokens,
      "workload.speculative.maxAdditionalTokens",
      add,
    );
  }
  validateNonNegativeInteger(
    scenario.execution.topologyEpoch,
    "execution.topologyEpoch",
    add,
  );
  validateNonNegativeInteger(scenario.execution.seed, "execution.seed", add);
  validatePositiveInteger(
    scenario.execution.maxEvents,
    "execution.maxEvents",
    add,
  );
  if (scenario.execution.features === undefined) {
    add(
      "invalid_feature",
      "execution.features",
      "must declare resource-manager features",
    );
  } else if (typeof scenario.execution.features.ssdStreaming !== "boolean") {
    add(
      "invalid_feature",
      "execution.features.ssdStreaming",
      "must be boolean",
    );
  }
  for (const key of ["tensor", "pipeline", "expert", "data"] as const) {
    const value = scenario.execution.parallelism[key];
    validatePositiveInteger(value, `execution.parallelism.${key}`, add);
  }

  const domains = indexUnique(
    scenario.memoryDomains,
    "memoryDomains",
    issues,
  );
  const devices = indexUnique(scenario.devices, "devices", issues);
  const links = indexUnique(scenario.links, "links", issues);
  indexUnique(scenario.placements, "placements", issues, "partitionId");
  indexUnique(scenario.transfers, "transfers", issues);
  const groups = indexUnique(scenario.groups, "groups", issues);

  for (const [index, domain] of scenario.memoryDomains.entries()) {
    const path = `memoryDomains[${index}]`;
    validateEnum(domain.kind, MEMORY_DOMAIN_KINDS, `${path}.kind`, add);
    validateProvenance(domain.provenance, `${path}.provenance`, add);
    validatePositiveInteger(domain.capacityBytes, `${path}.capacityBytes`, add);
    validatePositiveInteger(
      domain.resourceLimitBytes,
      `${path}.resourceLimitBytes`,
      add,
    );
    if (domain.resourceLimitBytes > domain.capacityBytes) {
      add(
        "resource_limit",
        `${path}.resourceLimitBytes`,
        `resource limit ${domain.resourceLimitBytes} exceeds physical capacity ${domain.capacityBytes}`,
      );
    }
    validatePositiveInteger(
      domain.bandwidthBytesPerSec,
      `${path}.bandwidthBytesPerSec`,
      add,
    );
    validateNonNegativeInteger(domain.latencyNs, `${path}.latencyNs`, add);
    validateUniqueStrings(
      domain.allocationClasses,
      `${path}.allocationClasses`,
      add,
    );
    for (const [classIndex, allocationClass] of
      domain.allocationClasses.entries()) {
      validateEnum(
        allocationClass,
        ALLOCATION_CLASSES,
        `${path}.allocationClasses[${classIndex}]`,
        add,
      );
    }
    validateUniqueStrings(domain.accessibleBy, `${path}.accessibleBy`, add);
    if (domain.allocationClasses.length === 0) {
      add("empty_classes", `${path}.allocationClasses`, "must not be empty");
    }
    for (const deviceId of domain.accessibleBy) {
      if (!devices.has(deviceId)) {
        add(
          "unknown_device",
          `${path}.accessibleBy`,
          `unknown device ${deviceId}`,
        );
      }
    }
    validateGovernor(domain, devices, path, add);
    if (
      domain.kind === "unified"
      && domain.allocationClasses.includes("device")
    ) {
      add(
        "unified_class",
        `${path}.allocationClasses`,
        "unified domains must use the unified allocation class",
      );
    }
    if (domain.kind === "storage") {
      if (
        domain.allocationClasses.length !== 1
        || domain.allocationClasses[0] !== "storage"
      ) {
        add(
          "storage_class",
          `${path}.allocationClasses`,
          "storage domains must exclusively use the storage allocation class",
        );
      }
      if (domain.governor.kind !== "none") {
        add(
          "storage_governor",
          `${path}.governor`,
          "storage domains must not use host or device memory governors",
        );
      }
      if (domain.accessibleBy.length === 0) {
        add(
          "storage_access",
          `${path}.accessibleBy`,
          "storage domains require at least one local CPU endpoint",
        );
      }
      for (const deviceId of domain.accessibleBy) {
        const device = devices.get(deviceId);
        if (
          device !== undefined
          && (device.kind !== "cpu" || device.nodeId !== domain.nodeId)
        ) {
          add(
            "storage_access",
            `${path}.accessibleBy`,
            `${deviceId} must be a CPU on storage node ${domain.nodeId}`,
          );
        }
      }
    } else if (domain.allocationClasses.includes("storage")) {
      add(
        "storage_class",
        `${path}.allocationClasses`,
        "non-storage domains cannot use the storage allocation class",
      );
    }
  }

  for (const [index, device] of scenario.devices.entries()) {
    const path = `devices[${index}]`;
    validateEnum(device.kind, DEVICE_KINDS, `${path}.kind`, add);
    validateProvenance(device.provenance, `${path}.provenance`, add);
    if (device.id.length === 0 || device.nodeId.length === 0) {
      add("empty_id", path, "device id and nodeId must not be empty");
    }
    if (device.executionProvider.length === 0) {
      add("empty_ep", `${path}.executionProvider`, "must not be empty");
    }
    validatePositiveInteger(
      device.maxConcurrentCompute,
      `${path}.maxConcurrentCompute`,
      add,
    );
    validateUniqueStrings(
      device.memoryDomainIds,
      `${path}.memoryDomainIds`,
      add,
    );
    validateUniqueStrings(device.capabilities, `${path}.capabilities`, add);
    for (const [capabilityIndex, capability] of
      device.capabilities.entries()) {
      validateEnum(
        capability,
        COMPUTE_CAPABILITIES,
        `${path}.capabilities[${capabilityIndex}]`,
        add,
      );
    }
    validateUniqueStrings(
      device.supportedDtypes,
      `${path}.supportedDtypes`,
      add,
    );
    if (device.memoryDomainIds.length === 0) {
      add("no_memory", `${path}.memoryDomainIds`, "must not be empty");
    }
    for (const domainId of device.memoryDomainIds) {
      const domain = domains.get(domainId);
      if (!domain) {
        add(
          "unknown_domain",
          `${path}.memoryDomainIds`,
          `unknown domain ${domainId}`,
        );
      } else if (!domain.accessibleBy.includes(device.id)) {
        add(
          "asymmetric_access",
          `${path}.memoryDomainIds`,
          `${domainId} does not grant access to ${device.id}`,
        );
      }
    }
  }

  for (const [index, link] of scenario.links.entries()) {
    const path = `links[${index}]`;
    validateEnum(link.kind, LINK_KINDS, `${path}.kind`, add);
    validateProvenance(link.provenance, `${path}.provenance`, add);
    if (!domains.has(link.sourceDomainId)) {
      add(
        "unknown_domain",
        `${path}.sourceDomainId`,
        `unknown domain ${link.sourceDomainId}`,
      );
    }
    if (!domains.has(link.targetDomainId)) {
      add(
        "unknown_domain",
        `${path}.targetDomainId`,
        `unknown domain ${link.targetDomainId}`,
      );
    }
    if (link.sourceDomainId === link.targetDomainId) {
      add("self_link", path, "link endpoints must differ");
    }
    validatePositiveInteger(
      link.bandwidthBytesPerSec,
      `${path}.bandwidthBytesPerSec`,
      add,
    );
    validateNonNegativeInteger(link.latencyNs, `${path}.latencyNs`, add);
    validatePositiveInteger(
      link.concurrencyLanes,
      `${path}.concurrencyLanes`,
      add,
    );
  }

  const allocationOwners = new Map<string, string>();
  const allocationDomains = new Map<string, string>();
  const allocationReservations = new Map<
    string,
    AllocationReservation
  >();
  const chargedBytes = new Map<string, number>();
  for (const [index, placement] of scenario.placements.entries()) {
    const path = `placements[${index}]`;
    const device = devices.get(placement.deviceId);
    if (!device) {
      add(
        "unknown_device",
        `${path}.deviceId`,
        `unknown device ${placement.deviceId}`,
      );
    } else {
      for (const capability of placement.requiredCapabilities) {
        if (!device.capabilities.includes(capability)) {
          add(
            "missing_capability",
            `${path}.requiredCapabilities`,
            `${placement.deviceId} lacks ${capability}`,
          );
        }
      }
    }
    validateUniqueStrings(
      placement.requiredCapabilities,
      `${path}.requiredCapabilities`,
      add,
    );
    for (const [capabilityIndex, capability] of
      placement.requiredCapabilities.entries()) {
      validateEnum(
        capability,
        COMPUTE_CAPABILITIES,
        `${path}.requiredCapabilities[${capabilityIndex}]`,
        add,
      );
    }

    for (const [allocationIndex, allocation] of placement.allocations.entries()) {
      const allocationPath = `${path}.allocations[${allocationIndex}]`;
      validateEnum(
        allocation.allocationClass,
        ALLOCATION_CLASSES,
        `${allocationPath}.allocationClass`,
        add,
      );
      validateEnum(
        allocation.purpose,
        ALLOCATION_PURPOSES,
        `${allocationPath}.purpose`,
        add,
      );
      validatePositiveInteger(allocation.bytes, `${allocationPath}.bytes`, add);
      const previousOwner = allocationOwners.get(allocation.physicalAllocationId);
      if (allocation.physicalAllocationId.length === 0 || previousOwner) {
        add(
          "duplicate_allocation",
          `${allocationPath}.physicalAllocationId`,
          previousOwner
            ? `already owned by ${previousOwner}`
            : "must not be empty",
        );
      } else {
        allocationOwners.set(
          allocation.physicalAllocationId,
          placement.partitionId,
        );
        allocationDomains.set(
          allocation.physicalAllocationId,
          allocation.domainId,
        );
        allocationReservations.set(
          allocation.physicalAllocationId,
          allocation,
        );
      }

      const domain = domains.get(allocation.domainId);
      if (!domain) {
        add(
          "unknown_domain",
          `${allocationPath}.domainId`,
          `unknown domain ${allocation.domainId}`,
        );
        continue;
      }
      if (!domain.allocationClasses.includes(allocation.allocationClass)) {
        add(
          "unsupported_allocation_class",
          `${allocationPath}.allocationClass`,
          `${allocation.domainId} does not support ${allocation.allocationClass}`,
        );
      }
      if (device && !device.memoryDomainIds.includes(allocation.domainId)) {
        add(
          "inaccessible_allocation",
          allocationPath,
          `${placement.deviceId} cannot access ${allocation.domainId}`,
        );
      }
      const current = chargedBytes.get(allocation.domainId) ?? 0;
      const total = checkedAddForValidation(
        current,
        allocation.bytes,
        `${allocationPath}.bytes`,
        add,
      );
      if (total !== undefined) {
        chargedBytes.set(allocation.domainId, total);
      }
    }
  }

  for (const [index, placement] of scenario.placements.entries()) {
    const path = `placements[${index}].sharedAllocationIds`;
    const sharedAllocationIds = placement.sharedAllocationIds ?? [];
    validateUniqueStrings(sharedAllocationIds, path, add);
    const owned = new Set(
      placement.allocations.map((allocation) => allocation.physicalAllocationId),
    );
    const device = devices.get(placement.deviceId);
    for (const allocationId of sharedAllocationIds) {
      if (owned.has(allocationId)) {
        add(
          "self_alias",
          path,
          `${allocationId} is already owned by ${placement.partitionId}`,
        );
        continue;
      }
      const domainId = allocationDomains.get(allocationId);
      if (!domainId) {
        add(
          "unknown_allocation",
          path,
          `unknown shared allocation ${allocationId}`,
        );
      } else if (device && !device.memoryDomainIds.includes(domainId)) {
        add(
          "inaccessible_allocation",
          path,
          `${placement.deviceId} cannot access shared allocation ${allocationId} in ${domainId}`,
        );
      }
    }
  }

  for (const [domainId, bytes] of chargedBytes) {
    const domain = domains.get(domainId);
    if (domain && bytes > domain.capacityBytes) {
      add(
        "domain_over_capacity",
        `memoryDomains.${domainId}`,
        `reservations ${bytes} exceed capacity ${domain.capacityBytes}`,
      );
    }
  }

  for (const [index, transfer] of scenario.transfers.entries()) {
    validateTransfer(
      transfer,
      index,
      domains,
      scenario.links,
      allocationReservations,
      add,
    );
  }

  const rankOwners = new Map<string, string>();
  for (const [index, group] of scenario.groups.entries()) {
    const path = `groups[${index}]`;
    if (group.orderedRanks.length === 0) {
      add("empty_group", `${path}.orderedRanks`, "must not be empty");
    }
    const localRanks = new Set<string>();
    const localDevices = new Set<string>();
    for (const [rankIndex, rank] of group.orderedRanks.entries()) {
      const rankPath = `${path}.orderedRanks[${rankIndex}]`;
      if (rank.rankId.length === 0 || localRanks.has(rank.rankId)) {
        add("duplicate_rank", `${rankPath}.rankId`, `invalid rank ${rank.rankId}`);
      }
      localRanks.add(rank.rankId);
      const priorDevice = rankOwners.get(rank.rankId);
      if (priorDevice && priorDevice !== rank.deviceId) {
        add(
          "rank_remap",
          `${rankPath}.rankId`,
          `${rank.rankId} maps to both ${priorDevice} and ${rank.deviceId}`,
        );
      } else {
        rankOwners.set(rank.rankId, rank.deviceId);
      }
      if (!devices.has(rank.deviceId)) {
        add(
          "unknown_device",
          `${rankPath}.deviceId`,
          `unknown device ${rank.deviceId}`,
        );
      }
      if (localDevices.has(rank.deviceId)) {
        add(
          "duplicate_group_device",
          `${rankPath}.deviceId`,
          `${rank.deviceId} appears twice in ${group.id}`,
        );
      }
      localDevices.add(rank.deviceId);
    }
  }

  const rankCount = new Set(rankOwners.keys()).size;
  const parallelism = scenario.execution.parallelism;
  if (
    parallelism.composition !== "cartesian"
    && parallelism.composition !== "overlap_by_capability"
  ) {
    add(
      "parallelism_composition",
      "execution.parallelism.composition",
      `unsupported composition ${String(parallelism.composition)}`,
    );
  }
  if (
    parallelism.composition === "overlap_by_capability"
    && (parallelism.pipeline !== 1 || parallelism.data !== 1)
  ) {
    add(
      "parallelism_composition",
      "execution.parallelism",
      "overlap_by_capability currently requires pipeline and data degrees of one",
    );
  }
  if (
    parallelism.composition === "overlap_by_capability"
    && parallelism.tensor !== parallelism.expert
  ) {
    add(
      "parallelism_composition",
      "execution.parallelism",
      "overlap_by_capability currently requires equal tensor and expert degrees",
    );
  }
  if (
    parallelism.composition === "overlap_by_capability"
    && !scenario.groups.some(
      (group) => group.orderedRanks.length === parallelism.tensor,
    )
  ) {
    add(
      "parallelism_composition",
      "groups",
      `overlap_by_capability requires a ${parallelism.tensor}-rank communicator`,
    );
  }
  const requiredRanks = parallelism.composition === "overlap_by_capability"
    ? Math.max(parallelism.tensor, parallelism.expert)
    : checkedProductForValidation(
        [
          parallelism.tensor,
          parallelism.pipeline,
          parallelism.expert,
          parallelism.data,
        ],
        "execution.parallelism",
        add,
      );
  if (requiredRanks !== undefined && requiredRanks > Math.max(rankCount, 1)) {
    add(
      "parallelism_participants",
      "execution.parallelism",
      `requires ${requiredRanks} ranks but only ${rankCount} are declared`,
    );
  }

  for (const [index, coefficient] of scenario.calibration.coefficients.entries()) {
    validateProvenance(
      coefficient.provenance,
      `calibration.coefficients[${index}].provenance`,
      add,
    );
    if (
      coefficient.id.length === 0
      || coefficient.unit.length === 0
      || !Number.isFinite(coefficient.value)
      || coefficient.value < 0
    ) {
      add(
        "invalid_calibration",
        `calibration.coefficients[${index}]`,
        "id/unit must be non-empty and value must be finite and non-negative",
      );
    }
  }

  // Access these maps so duplicate-id diagnostics cover every collection even
  // when no later object references them.
  void links;
  void groups;

  return { valid: issues.length === 0, issues };
}

export function findTransferPath(
  scenario: Pick<SimulationScenario, "memoryDomains" | "links">,
  transfer: TransferRequirement,
): readonly string[] | undefined {
  return findTransferRoute(scenario, transfer)?.domainIds;
}

export function findTransferRoute(
  scenario: Pick<SimulationScenario, "memoryDomains" | "links">,
  transfer: TransferRequirement,
): TransferRoute | undefined {
  if (!Number.isSafeInteger(transfer.bytes) || transfer.bytes <= 0) {
    return undefined;
  }
  if (transfer.sourceDomainId === transfer.targetDomainId) {
    if (transfer.requiresPinnedStaging) {
      return undefined;
    }
    return {
      domainIds: [transfer.sourceDomainId],
      linkIds: [],
      declaredDurationNs: 0,
    };
  }

  const domains = new Map(
    scenario.memoryDomains.map((domain) => [domain.id, domain]),
  );
  if (!domains.has(transfer.sourceDomainId) || !domains.has(transfer.targetDomainId)) {
    return undefined;
  }

  const adjacency = new Map<string, SimLinkSpec[]>();
  for (const link of scenario.links) {
    const outgoing = adjacency.get(link.sourceDomainId) ?? [];
    outgoing.push(link);
    adjacency.set(link.sourceDomainId, outgoing);
  }
  for (const links of adjacency.values()) {
    links.sort((left, right) => compareRouteKeys(left.id, right.id));
  }

  interface Candidate {
    readonly domainId: string;
    readonly domainIds: readonly string[];
    readonly linkIds: readonly string[];
    readonly routeKey: string;
    readonly declaredDurationNs: number;
    readonly hasPinnedIntermediate: boolean;
    readonly visitedDomainIds: ReadonlySet<string>;
  }
  const initial: Candidate = {
    domainId: transfer.sourceDomainId,
    domainIds: [transfer.sourceDomainId],
    linkIds: [],
    routeKey: "",
    declaredDurationNs: 0,
    hasPinnedIntermediate: false,
    visitedDomainIds: new Set([transfer.sourceDomainId]),
  };
  const queue: Candidate[] = [initial];
  const labelsByState = new Map<string, Candidate[]>([[
    routeStateKey(initial.domainId, initial.hasPinnedIntermediate),
    [initial],
  ]]);
  while (queue.length > 0) {
    queue.sort(compareRouteCandidates);
    const candidate = queue.shift();
    if (!candidate) {
      break;
    }
    const stateKey = routeStateKey(
      candidate.domainId,
      candidate.hasPinnedIntermediate,
    );
    const labels = labelsByState.get(stateKey);
    if (!labels?.some((label) => (
      label.declaredDurationNs === candidate.declaredDurationNs
      && label.routeKey === candidate.routeKey
    ))) {
      continue;
    }
    if (candidate.domainId === transfer.targetDomainId) {
      if (
        !transfer.requiresPinnedStaging
        || candidate.hasPinnedIntermediate
      ) {
        return {
          domainIds: candidate.domainIds,
          linkIds: candidate.linkIds,
          declaredDurationNs: candidate.declaredDurationNs,
        };
      }
      continue;
    }
    for (const link of adjacency.get(candidate.domainId) ?? []) {
      if (
        !domains.has(link.targetDomainId)
        || candidate.visitedDomainIds.has(link.targetDomainId)
      ) {
        continue;
      }
      const edgeDurationNs = declaredLinkDurationNs(link, transfer.bytes);
      if (edgeDurationNs === undefined) {
        continue;
      }
      const currentIsIntermediate =
        candidate.domainId !== transfer.sourceDomainId;
      const nextHasPinnedIntermediate =
        candidate.hasPinnedIntermediate || (
        currentIsIntermediate
        && domains.get(candidate.domainId)?.allocationClasses.includes(
          "pinned",
        ) === true
      );
      const declaredDurationNs =
        candidate.declaredDurationNs + edgeDurationNs;
      if (!Number.isSafeInteger(declaredDurationNs)) {
        continue;
      }
      const linkIds = [...candidate.linkIds, link.id];
      const routeKey = JSON.stringify(linkIds);
      const nextStateKey = routeStateKey(
        link.targetDomainId,
        nextHasPinnedIntermediate,
      );
      const visitedDomainIds = new Set(candidate.visitedDomainIds);
      visitedDomainIds.add(link.targetDomainId);
      const next: Candidate = {
        domainId: link.targetDomainId,
        domainIds: [...candidate.domainIds, link.targetDomainId],
        linkIds,
        routeKey,
        declaredDurationNs,
        hasPinnedIntermediate: nextHasPinnedIntermediate,
        visitedDomainIds,
      };
      const priorLabels = labelsByState.get(nextStateKey) ?? [];
      if (priorLabels.some((prior) => routeLabelDominates(prior, next))) {
        continue;
      }
      labelsByState.set(
        nextStateKey,
        [
          ...priorLabels.filter((prior) => (
            !routeLabelDominates(next, prior)
          )),
          next,
        ],
      );
      queue.push(next);
    }
  }
  return undefined;
}

function routeStateKey(
  domainId: string,
  hasPinnedIntermediate: boolean,
): string {
  return JSON.stringify([domainId, hasPinnedIntermediate]);
}

function compareRouteCandidates(
  left: {
    readonly declaredDurationNs: number;
    readonly routeKey: string;
  },
  right: {
    readonly declaredDurationNs: number;
    readonly routeKey: string;
  },
): number {
  if (left.declaredDurationNs !== right.declaredDurationNs) {
    return left.declaredDurationNs < right.declaredDurationNs ? -1 : 1;
  }
  return compareRouteKeys(left.routeKey, right.routeKey);
}

function routeLabelDominates(
  left: {
    readonly declaredDurationNs: number;
    readonly routeKey: string;
    readonly visitedDomainIds: ReadonlySet<string>;
  },
  right: {
    readonly declaredDurationNs: number;
    readonly routeKey: string;
    readonly visitedDomainIds: ReadonlySet<string>;
  },
): boolean {
  if (
    left.declaredDurationNs > right.declaredDurationNs
    || (
      left.declaredDurationNs === right.declaredDurationNs
      && compareRouteKeys(left.routeKey, right.routeKey) > 0
    )
  ) {
    return false;
  }
  for (const domainId of left.visitedDomainIds) {
    if (!right.visitedDomainIds.has(domainId)) {
      return false;
    }
  }
  return true;
}

function compareRouteKeys(left: string, right: string): number {
  return left === right ? 0 : left < right ? -1 : 1;
}

function declaredLinkDurationNs(
  link: SimLinkSpec,
  bytes: number,
): number | undefined {
  if (
    !Number.isSafeInteger(link.latencyNs)
    || link.latencyNs < 0
    || !Number.isSafeInteger(link.bandwidthBytesPerSec)
    || link.bandwidthBytesPerSec <= 0
  ) {
    return undefined;
  }
  const serviceNs = Math.ceil(
    bytes / link.bandwidthBytesPerSec * 1_000_000_000,
  );
  const durationNs = link.latencyNs + serviceNs;
  return Number.isSafeInteger(durationNs) ? durationNs : undefined;
}

function validateTransfer(
  transfer: TransferRequirement,
  index: number,
  domains: ReadonlyMap<string, MemoryDomainSpec>,
  links: readonly SimLinkSpec[],
  allocations: ReadonlyMap<
    string,
    AllocationReservation
  >,
  add: (code: string, path: string, message: string) => void,
): void {
  const path = `transfers[${index}]`;
  validatePositiveInteger(transfer.bytes, `${path}.bytes`, add);
  if (!domains.has(transfer.sourceDomainId)) {
    add(
      "unknown_domain",
      `${path}.sourceDomainId`,
      `unknown domain ${transfer.sourceDomainId}`,
    );
  }
  if (!domains.has(transfer.targetDomainId)) {
    add(
      "unknown_domain",
      `${path}.targetDomainId`,
      `unknown domain ${transfer.targetDomainId}`,
    );
  }
  const transferPath = findTransferPath(
    { memoryDomains: [...domains.values()], links },
    transfer,
  );
  if (!transferPath) {
    add(
      "no_transfer_path",
      path,
      transfer.requiresPinnedStaging
        ? "no directed path with a pinned intermediate domain"
        : "no directed transfer path",
    );
    return;
  }

  validateUniqueStrings(
    transfer.stagingAllocationIds,
    `${path}.stagingAllocationIds`,
    add,
  );
  if (!transfer.requiresPinnedStaging && transfer.stagingAllocationIds.length > 0) {
    add(
      "unexpected_staging",
      `${path}.stagingAllocationIds`,
      "staging allocations were supplied for a transfer that does not require them",
    );
  }
  if (transfer.requiresPinnedStaging) {
    const intermediateDomains = new Set(transferPath.slice(1, -1));
    const coveredDomains = new Set<string>();
    for (const allocationId of transfer.stagingAllocationIds) {
      const allocation = allocations.get(allocationId);
      if (!allocation) {
        add(
          "unknown_allocation",
          `${path}.stagingAllocationIds`,
          `unknown staging allocation ${allocationId}`,
        );
        continue;
      }
      const domain = domains.get(allocation.domainId);
      if (
        allocation.purpose !== "staging"
        || allocation.allocationClass !== "pinned"
        || allocation.bytes < transfer.bytes
        || !intermediateDomains.has(allocation.domainId)
        || domain?.allocationClasses.includes("pinned") !== true
      ) {
        add(
          "invalid_staging",
          `${path}.stagingAllocationIds`,
          `${allocationId} is not a sufficiently large pinned staging allocation on the transfer path`,
        );
        continue;
      }
      coveredDomains.add(allocation.domainId);
    }
    for (const domainId of intermediateDomains) {
      if (
        domains.get(domainId)?.allocationClasses.includes("pinned") === true
        && !coveredDomains.has(domainId)
      ) {
        add(
          "missing_staging",
          `${path}.stagingAllocationIds`,
          `no staging allocation covers intermediate domain ${domainId}`,
        );
      }
    }
  }
}

function validateGovernor(
  domain: MemoryDomainSpec,
  devices: ReadonlyMap<string, { readonly id: string; readonly nodeId: string }>,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  const governor = domain.governor as unknown as Record<string, unknown>;
  if (governor.kind === "host") {
    if (governor.nodeId !== domain.nodeId) {
      add(
        "governor_node",
        `${path}.governor`,
        `host governor ${String(governor.nodeId)} does not own node ${domain.nodeId}`,
      );
    }
  } else if (governor.kind === "device") {
    const device = typeof governor.deviceId === "string"
      ? devices.get(governor.deviceId)
      : undefined;
    if (!device || device.nodeId !== domain.nodeId) {
      add(
        "governor_device",
        `${path}.governor`,
        `unknown or cross-node owner ${String(governor.deviceId)}`,
      );
    }
  } else if (governor.kind !== "none") {
    add(
      "enum_value",
      `${path}.governor.kind`,
      `must be one of host, device, none; got ${String(governor.kind)}`,
    );
  }
}

function indexUnique<T extends object>(
  values: readonly T[],
  path: string,
  issues: ScenarioValidationIssue[],
  idKey: keyof T = "id" as keyof T,
): Map<string, T> {
  const result = new Map<string, T>();
  values.forEach((value, index) => {
    const id = value[idKey];
    if (typeof id !== "string" || id.length === 0 || result.has(id)) {
      issues.push({
        code: "duplicate_id",
        path: `${path}[${index}].${String(idKey)}`,
        message: `id must be non-empty and unique; got ${String(id)}`,
      });
    } else {
      result.set(id, value);
    }
  });
  return result;
}

function validateUniqueStrings(
  values: readonly string[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (new Set(values).size !== values.length || values.some((value) => value.length === 0)) {
    add("duplicate_value", path, "values must be non-empty and unique");
  }
}

function validateEnum(
  value: string,
  allowed: readonly string[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!allowed.includes(value)) {
    add(
      "enum_value",
      path,
      `must be one of ${allowed.join(", ")}; got ${String(value)}`,
    );
  }
}

function validateProvenance(
  provenance: { readonly confidence: string; readonly source: string },
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  validateEnum(
    provenance.confidence,
    CONFIDENCE_CLASSES,
    `${path}.confidence`,
    add,
  );
  if (provenance.source.length === 0) {
    add("empty_provenance", `${path}.source`, "must not be empty");
  }
}

function validatePositiveInteger(
  value: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    add("positive_integer", path, `must be a positive safe integer; got ${value}`);
  }
}

function validateNonNegativeInteger(
  value: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    add(
      "non_negative_integer",
      path,
      `must be a non-negative safe integer; got ${value}`,
    );
  }
}

function checkedAddForValidation(
  left: number,
  right: number,
  path: string,
  add: (code: string, path: string, message: string) => void,
): number | undefined {
  const result = left + right;
  if (!Number.isSafeInteger(result)) {
    add("integer_overflow", path, "sum exceeds Number.MAX_SAFE_INTEGER");
    return undefined;
  }
  return result;
}

function checkedProductForValidation(
  values: readonly number[],
  path: string,
  add: (code: string, path: string, message: string) => void,
): number | undefined {
  let result = 1;
  for (const value of values) {
    result *= value;
    if (!Number.isSafeInteger(result)) {
      add("integer_overflow", path, "product exceeds Number.MAX_SAFE_INTEGER");
      return undefined;
    }
  }
  return result;
}

export function domainSupportsClass(
  domain: MemoryDomainSpec,
  allocationClass: AllocationClass,
): boolean {
  return domain.allocationClasses.includes(allocationClass);
}

export function calculateScenarioMemoryLedger(
  scenario: SimulationScenario,
  options: ScenarioMemoryLedgerOptions = {},
): readonly ScenarioMemoryLedgerEntry[] {
  assertValidScenario(scenario);
  const allocationIds = new Set(scenario.placements.flatMap(
    (placement) => placement.allocations.map(
      (allocation) => allocation.physicalAllocationId,
    ),
  ));
  for (const [allocationId, bytes] of Object.entries(
    options.allocationBytes ?? {},
  )) {
    if (!allocationIds.has(allocationId)) {
      throw new ScenarioValidationError([{
        code: "unknown_allocation",
        path: `allocationBytes.${allocationId}`,
        message: `unknown allocation ${allocationId}`,
      }]);
    }
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new ScenarioValidationError([{
        code: "invalid_allocation_override",
        path: `allocationBytes.${allocationId}`,
        message: "allocation override must be a non-negative safe integer",
      }]);
    }
  }
  const reserved = new Map<string, number>();
  for (const placement of scenario.placements) {
    for (const allocation of placement.allocations) {
      const allocationBytes = options.allocationBytes?.[
        allocation.physicalAllocationId
      ] ?? allocation.bytes;
      const current = reserved.get(allocation.domainId) ?? 0;
      const total = current + allocationBytes;
      if (!Number.isSafeInteger(total)) {
        throw new ScenarioValidationError([{
          code: "integer_overflow",
          path: `placements.${placement.partitionId}`,
          message: "reservation sum exceeds Number.MAX_SAFE_INTEGER",
        }]);
      }
      reserved.set(allocation.domainId, total);
    }
  }
  return scenario.memoryDomains
    .map((domain) => {
      const enabled = domain.kind !== "storage"
        || scenario.execution.features.ssdStreaming;
      const reservedBytes = enabled ? reserved.get(domain.id) ?? 0 : 0;
      const capacityBytes = enabled ? domain.resourceLimitBytes : 0;
      return {
        domainId: domain.id,
        enabled,
        physicalCapacityBytes: domain.capacityBytes,
        capacityBytes,
        reservedBytes,
        freeBytes: capacityBytes - reservedBytes,
      };
    })
    .sort((left, right) => left.domainId.localeCompare(right.domainId));
}
