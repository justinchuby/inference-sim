import { useEffect, useState } from "react";
import {
  Check,
  Cpu,
  Gauge,
  HardDrive,
  Link2,
  MemoryStick,
  Network,
} from "lucide-react";
import type {
  ComputeCapability,
  MemoryDomainSpec,
  MultiNodeLanOptions,
  NetworkResourceSpec,
  NetworkTransportMode,
  SimDeviceSpec,
  SimLinkSpec,
  SimulationScenario,
} from "@inference-sim/core";
import {
  HARDWARE_COMPUTE_PROFILES,
  configureSmallLanNetwork,
  denseHardwareComputePeak,
  hardwareComputeDtypes,
  hardwareComputeProfile,
} from "@inference-sim/core";
import { Badge } from "./components/ui/badge.js";
import { Button } from "./components/ui/button.js";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "./components/ui/dialog.js";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select.js";
import { Switch } from "./components/ui/switch.js";
import {
  bytesToGibibytes,
  bytesToGigabytesPerSecond,
  COMPUTE_CAPABILITIES,
  finalizeEditedTopology,
  gibibytesToBytes,
  gigabytesPerSecondToBytes,
  LINK_KINDS,
  materializeNetworkResources,
  NETWORK_TRANSPORTS,
} from "./topology-editor.js";
import TopologyGraph from "./TopologyGraph.js";

const CUSTOM_COMPUTE_DTYPES = [
  "fp32",
  "bf16",
  "fp16",
  "fp8",
  "int8",
  "int4",
  "int2",
  "int1",
] as const;

export default function TopologyEditorDialog({
  open,
  scenario,
  onOpenChange,
  onSave,
}: {
  readonly open: boolean;
  readonly scenario: SimulationScenario;
  readonly onOpenChange: (open: boolean) => void;
  readonly onSave: (scenario: SimulationScenario) => void;
}): React.JSX.Element {
  const [draft, setDraft] = useState(scenario);
  const [view, setView] = useState<
    "map" | "resources" | "devices" | "links"
  >("map");
  const [advancedNetwork, setAdvancedNetwork] = useState(false);
  const [showLinkDetails, setShowLinkDetails] = useState(false);
  const [error, setError] = useState<string | undefined>();

  useEffect(() => {
    if (open) {
      setDraft(scenario);
      setView("map");
      setAdvancedNetwork((scenario.networkResources?.length ?? 0) > 0);
      setShowLinkDetails(false);
      setError(undefined);
    }
  }, [open, scenario]);

  const updateDevice = (
    id: string,
    update: (device: SimDeviceSpec) => SimDeviceSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      devices: current.devices.map((device) => (
        device.id === id ? update(device) : device
      )),
    }));
  };
  const updateDomain = (
    id: string,
    update: (domain: MemoryDomainSpec) => MemoryDomainSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      memoryDomains: current.memoryDomains.map((domain) => (
        domain.id === id ? update(domain) : domain
      )),
    }));
  };
  const updateLink = (
    id: string,
    update: (link: SimLinkSpec) => SimLinkSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      links: current.links.map((link) => (
        link.id === id ? update(link) : link
      )),
    }));
  };
  const updateNetworkResource = (
    id: string,
    update: (resource: NetworkResourceSpec) => NetworkResourceSpec,
  ) => {
    setDraft((current) => ({
      ...current,
      networkResources: (current.networkResources ?? []).map((resource) => (
        resource.id === id ? update(resource) : resource
      )),
    }));
  };
  const systemCount = new Set([
    ...draft.devices.map((device) => device.nodeId),
    ...draft.memoryDomains.map((domain) => domain.nodeId),
  ]).size;
  const lanOptions = smallLanOptions(draft, advancedNetwork);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex w-[min(calc(100vw-2rem),960px)] max-h-[min(88vh,820px)] flex-col">
        <div className="border-b border-zinc-200 px-5 py-4 pr-14">
          <DialogTitle className="text-base font-bold">
            Device topology
          </DialogTitle>
          <DialogDescription className="mt-1 text-xs text-zinc-500">
            {draft.id} · {systemCount} system{systemCount === 1 ? "" : "s"} ·{" "}
            {draft.devices.length} compute chips · {draft.links.length} directed links
          </DialogDescription>
        </div>

        <div className="flex min-h-0 flex-1 flex-col">
          <div className="flex gap-1 border-b border-zinc-200 px-5 py-2">
            <Button
              type="button"
              variant={view === "map" ? "secondary" : "ghost"}
              onClick={() => setView("map")}
            >
              <Network className="size-4" />
              Map
            </Button>
            <Button
              type="button"
              variant={view === "resources" ? "secondary" : "ghost"}
              onClick={() => setView("resources")}
            >
              <Gauge className="size-4" />
              Resources
            </Button>
            <Button
              type="button"
              variant={view === "devices" ? "secondary" : "ghost"}
              onClick={() => setView("devices")}
            >
              <Cpu className="size-4" />
              Devices
            </Button>
            <Button
              type="button"
              variant={view === "links" ? "secondary" : "ghost"}
              onClick={() => setView("links")}
            >
              <Link2 className="size-4" />
              Network
            </Button>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto bg-zinc-50 p-4 sm:p-5">
            {view === "map"
              ? (
                  <TopologyGraph
                    scenario={draft}
                    className="h-[min(65vh,620px)]"
                  />
                )
              : view === "resources"
              ? (
                  <ResourceManagerEditor
                    scenario={draft}
                    onSsdStreamingChange={(ssdStreaming) => {
                      setDraft((current) => ({
                        ...current,
                        execution: {
                          ...current.execution,
                          features: {
                            ...current.execution.features,
                            ssdStreaming,
                          },
                        },
                      }));
                    }}
                    onDomainChange={updateDomain}
                  />
                )
              : view === "devices"
              ? (
                  <div className="grid gap-3 xl:grid-cols-2">
                    {draft.devices.map((device) => (
                      <DeviceEditor
                        key={device.id}
                        device={device}
                        domains={device.memoryDomainIds.flatMap((id) => {
                          const domain = draft.memoryDomains.find(
                            (candidate) => candidate.id === id,
                          );
                          return domain === undefined ? [] : [domain];
                        })}
                        onDeviceChange={(update) => (
                          updateDevice(device.id, update)
                        )}
                        onDomainChange={updateDomain}
                      />
                    ))}
                  </div>
                )
              : (
                  <div className="space-y-4">
                    <section className="border border-zinc-200 bg-white p-4">
                      <SmallLanEditor
                        scenario={draft}
                        advanced={advancedNetwork}
                        onChange={(options) => {
                          try {
                            const configured = configureSmallLanNetwork(
                              draft,
                              options,
                            );
                            setDraft(configured);
                            setAdvancedNetwork(options.advanced ?? false);
                            setError(undefined);
                          } catch (networkError) {
                            setError(
                              networkError instanceof Error
                                ? networkError.message
                                : String(networkError),
                            );
                          }
                        }}
                      />
                    </section>

                    {advancedNetwork
                      ? (
                          <NetworkResourceEditor
                            resources={draft.networkResources ?? []}
                            domains={draft.memoryDomains}
                            onMaterialize={() => setDraft(
                              materializeNetworkResources,
                            )}
                            onChange={updateNetworkResource}
                          />
                        )
                      : null}

                    {lanOptions !== undefined
                      ? (
                          <div className="flex justify-end">
                            <Button
                              type="button"
                              variant="ghost"
                              onClick={() => setShowLinkDetails((shown) => (
                                !shown
                              ))}
                            >
                              <Link2 className="size-4" />
                              {showLinkDetails ? "Hide" : "Show"} directed links
                            </Button>
                          </div>
                        )
                      : null}
                    {lanOptions === undefined || showLinkDetails
                      ? (
                          <div className="grid gap-3 xl:grid-cols-2">
                            {draft.links.map((link) => (
                              <LinkEditor
                                key={link.id}
                                link={link}
                                domains={draft.memoryDomains}
                                networkResources={draft.networkResources ?? []}
                                advanced={advancedNetwork}
                                onChange={(update) => updateLink(link.id, update)}
                              />
                            ))}
                          </div>
                        )
                      : null}
                  </div>
                )}
          </div>
        </div>

        <div className="border-t border-zinc-200 bg-white px-5 py-3">
          {error
            ? <div className="mb-2 text-xs text-rose-700">{error}</div>
            : null}
          <div className="flex justify-end gap-2">
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => {
                try {
                  const edited = finalizeEditedTopology(draft);
                  onSave(edited);
                  onOpenChange(false);
                } catch (saveError) {
                  setError(
                    saveError instanceof Error
                      ? saveError.message
                      : String(saveError),
                  );
                }
              }}
            >
              <Check className="size-4" />
              Apply topology
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function DeviceEditor({
  device,
  domains,
  onDeviceChange,
  onDomainChange,
}: {
  readonly device: SimDeviceSpec;
  readonly domains: readonly MemoryDomainSpec[];
  readonly onDeviceChange: (
    update: (device: SimDeviceSpec) => SimDeviceSpec,
  ) => void;
  readonly onDomainChange: (
    id: string,
    update: (domain: MemoryDomainSpec) => MemoryDomainSpec,
  ) => void;
}): React.JSX.Element {
  const computeProfiles = HARDWARE_COMPUTE_PROFILES.filter((profile) => (
    profile.deviceKind === device.kind
  ));
  const computeProfileVendors = [...new Set(computeProfiles.map((profile) => (
    profile.vendor
  )))];
  const selectedProfile = hardwareComputeProfile(device.computeProfileId);
  const customCompute = device.customComputePeaks !== undefined;
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-bold">{device.id}</h3>
          <div className="truncate text-[11px] text-zinc-500">
            {device.nodeId}
          </div>
        </div>
        <Badge variant="neutral">{device.kind}</Badge>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <EditorField label="Hardware compute profile" className="sm:col-span-2">
          <Select
            value={device.computeProfileId
              ?? (customCompute ? "__custom__" : "__unspecified__")}
            onValueChange={(value) => onDeviceChange((current) => {
              if (value === "__unspecified__") {
                const { computeProfileId: _removed, ...rest } = current;
                const { customComputePeaks: _customRemoved, ...withoutCustom } = rest;
                return withoutCustom;
              }
              if (value === "__custom__") {
                const { computeProfileId: _removed, ...rest } = current;
                return {
                  ...rest,
                  customComputePeaks: current.customComputePeaks ?? [{
                    dtype: current.supportedDtypes[0] ?? "fp32",
                    operationsPerSecond: 1e12,
                  }],
                };
              }
              const profile = hardwareComputeProfile(value);
              const dtypes = profile === undefined
                ? []
                : hardwareComputeDtypes(profile);
              const { customComputePeaks: _customRemoved, ...withoutCustom } = current;
              return {
                ...withoutCustom,
                computeProfileId: value,
                supportedDtypes: dtypes.length === 0
                  ? current.supportedDtypes
                  : dtypes,
              };
            })}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="__unspecified__">Unspecified hardware</SelectItem>
              <SelectItem value="__custom__">Custom dense peaks</SelectItem>
              {computeProfileVendors.map((vendor, vendorIndex) => (
                <SelectGroup key={vendor}>
                  {vendorIndex === 0 ? null : <SelectSeparator />}
                  <SelectLabel>{vendor}</SelectLabel>
                  {computeProfiles.filter((profile) => (
                    profile.vendor === vendor
                  )).map((profile) => (
                    <SelectItem key={profile.id} value={profile.id}>
                      {profile.model}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
          {customCompute ? (
            <div className="mt-3 border-t border-zinc-200 pt-3">
              <div className="mb-2 text-[11px] font-semibold text-zinc-600">
                Custom dense peaks · TOPS
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {CUSTOM_COMPUTE_DTYPES.map((dtype) => (
                  <EditorField key={dtype} label={dtype.toUpperCase()}>
                    <NumberInput
                      value={(device.customComputePeaks?.find((peak) => (
                        peak.dtype === dtype
                      ))?.operationsPerSecond ?? 0) / 1e12}
                      minimum={0}
                      step={0.1}
                      onChange={(tops) => onDeviceChange((current) => (
                        updateCustomComputePeak(current, dtype, tops)
                      ))}
                    />
                  </EditorField>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-4 text-zinc-500">
                Zero leaves a dtype unbound. Positive values become user-declared dense roofline ceilings.
              </p>
            </div>
          ) : selectedProfile === undefined ? (
            <p className="mt-1.5 text-[11px] leading-4 text-zinc-500">
              No official peak is bound. The simulator may use a calibrated or heuristic effective ceiling.
            </p>
          ) : (
            <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] leading-4 text-zinc-500">
              <span>{computeProfileSummary(selectedProfile)}</span>
              {selectedProfile.sources[0] === undefined ? null : (
                <a
                  className="font-semibold text-sky-700 hover:underline"
                  href={selectedProfile.sources[0].url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Official source
                </a>
              )}
            </div>
          )}
        </EditorField>
        <EditorField label="Execution provider">
          <TextInput
            value={device.executionProvider}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              executionProvider: value,
            }))}
          />
        </EditorField>
        <EditorField label="Compute lanes">
          <NumberInput
            value={device.maxConcurrentCompute}
            minimum={1}
            step={1}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              maxConcurrentCompute: value,
            }))}
          />
        </EditorField>
        <EditorField label="Supported dtypes" className="sm:col-span-2">
          <TextInput
            value={device.supportedDtypes.join(", ")}
            onChange={(value) => onDeviceChange((current) => ({
              ...current,
              supportedDtypes: splitList(value),
            }))}
          />
        </EditorField>
      </div>

      <div className="mt-4 border-t border-zinc-200 pt-3">
        <div className="mb-2 text-xs font-semibold text-zinc-600">
          Capabilities
        </div>
        <div className="flex flex-wrap gap-2">
          {COMPUTE_CAPABILITIES.map((capability) => {
            const active = device.capabilities.includes(capability);
            return (
              <label
                key={capability}
                className="flex items-center gap-1.5 text-xs text-zinc-700"
              >
                <input
                  type="checkbox"
                  checked={active}
                  onChange={() => onDeviceChange((current) => ({
                    ...current,
                    capabilities: toggleCapability(
                      current.capabilities,
                      capability,
                    ),
                  }))}
                />
                {capability}
              </label>
            );
          })}
        </div>
      </div>

      {domains.map((domain) => (
        <div key={domain.id} className="mt-4 border-t border-zinc-200 pt-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <div className="flex min-w-0 items-center gap-2">
              <MemoryStick className="size-4 shrink-0 text-zinc-500" />
              <span className="truncate text-xs font-semibold">
                {domain.id}
              </span>
            </div>
            <Badge variant="neutral">{domain.kind}</Badge>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            <EditorField label="Physical capacity GiB">
              <NumberInput
                value={bytesToGibibytes(domain.capacityBytes)}
                minimum={0.001}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  capacityBytes: gibibytesToBytes(value),
                  resourceLimitBytes: Math.min(
                    current.resourceLimitBytes,
                    gibibytesToBytes(value),
                  ),
                }))}
              />
            </EditorField>
            <EditorField label="Bandwidth GB/s">
              <NumberInput
                value={bytesToGigabytesPerSecond(
                  domain.bandwidthBytesPerSec,
                )}
                minimum={0.001}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  bandwidthBytesPerSec:
                    gigabytesPerSecondToBytes(value),
                }))}
              />
            </EditorField>
            <EditorField label="Latency ns">
              <NumberInput
                value={domain.latencyNs}
                minimum={0}
                step={1}
                onChange={(value) => onDomainChange(domain.id, (current) => ({
                  ...current,
                  latencyNs: value,
                }))}
              />
            </EditorField>
          </div>
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-600">Coherent</span>
            <Switch
              checked={domain.coherent}
              onCheckedChange={(coherent) => (
                onDomainChange(domain.id, (current) => ({
                  ...current,
                  coherent,
                }))
              )}
            />
          </div>
        </div>
      ))}
    </section>
  );
}

function updateCustomComputePeak(
  device: SimDeviceSpec,
  dtype: string,
  tops: number,
): SimDeviceSpec {
  const peaks = (device.customComputePeaks ?? []).filter((peak) => (
    peak.dtype !== dtype
  ));
  const nextPeaks = tops <= 0
    ? peaks
    : [...peaks, { dtype, operationsPerSecond: tops * 1e12 }];
  return {
    ...device,
    customComputePeaks: nextPeaks,
    supportedDtypes: tops <= 0 || device.supportedDtypes.includes(dtype)
      ? device.supportedDtypes
      : [...device.supportedDtypes, dtype],
  };
}

function computeProfileSummary(
  profile: (typeof HARDWARE_COMPUTE_PROFILES)[number],
): string {
  const dense = profile.peaks.filter((item) => item.sparsity === "dense");
  if (dense.length === 0) {
    return profile.peaks.length === 0
      ? "Vendor has not published an absolute dtype-specific peak."
      : "Published peaks exist, but dense semantics are not explicit.";
  }
  const dtypes = [...new Set(dense.map((item) => item.dtype))];
  return dtypes.map((dtype) => denseHardwareComputePeak(profile.id, dtype)!)
    .map((item) => (
    `${item.dtype.toUpperCase()} ${formatComputePeak(item.operationsPerSecond)}`
  )).join(" · ");
}

function formatComputePeak(operationsPerSecond: number): string {
  return operationsPerSecond >= 1e15
    ? `${(operationsPerSecond / 1e15).toLocaleString(undefined, { maximumFractionDigits: 3 })} POPS`
    : `${(operationsPerSecond / 1e12).toLocaleString(undefined, { maximumFractionDigits: 1 })} TOPS`;
}

function ResourceManagerEditor({
  scenario,
  onSsdStreamingChange,
  onDomainChange,
}: {
  readonly scenario: SimulationScenario;
  readonly onSsdStreamingChange: (enabled: boolean) => void;
  readonly onDomainChange: (
    id: string,
    update: (domain: MemoryDomainSpec) => MemoryDomainSpec,
  ) => void;
}): React.JSX.Element {
  return (
    <div className="space-y-4">
      <section className="border border-zinc-200 bg-white p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <HardDrive className="size-5 shrink-0 text-zinc-600" />
            <div className="min-w-0">
              <h3 className="text-sm font-bold">SSD streaming</h3>
              <p className="mt-0.5 text-xs text-zinc-500">
                Allow cold weights and background prefetches to read from SSD.
              </p>
            </div>
          </div>
          <Switch
            aria-label="SSD streaming"
            checked={scenario.execution.features.ssdStreaming}
            onCheckedChange={onSsdStreamingChange}
          />
        </div>
      </section>

      <div className="grid gap-3 xl:grid-cols-2">
        {scenario.memoryDomains.map((domain) => {
          const limitRatio = domain.resourceLimitBytes / domain.capacityBytes;
          const disabled = domain.kind === "storage"
            && !scenario.execution.features.ssdStreaming;
          return (
            <section
              key={domain.id}
              className="border border-zinc-200 bg-white p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="truncate text-sm font-bold">{domain.id}</h3>
                  <div className="mt-0.5 text-[11px] text-zinc-500">
                    {formatDomainKind(domain.kind)} ·{" "}
                    {bytesToGibibytes(domain.capacityBytes).toLocaleString(
                      "en-US",
                      { maximumFractionDigits: 2 },
                    )} GiB physical
                  </div>
                </div>
                <Badge variant={disabled ? "neutral" : "success"}>
                  {disabled ? "disabled" : `${Math.round(limitRatio * 100)}%`}
                </Badge>
              </div>
              <div className="mt-4">
                <EditorField label="Allocation limit GiB">
                  <NumberInput
                    value={bytesToGibibytes(domain.resourceLimitBytes)}
                    minimum={0.001}
                    maximum={bytesToGibibytes(domain.capacityBytes)}
                    step={1}
                    onChange={(value) => onDomainChange(
                      domain.id,
                      (current) => ({
                        ...current,
                        resourceLimitBytes: gibibytesToBytes(value),
                      }),
                    )}
                  />
                </EditorField>
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

function formatDomainKind(kind: MemoryDomainSpec["kind"]): string {
  switch (kind) {
    case "device":
      return "VRAM";
    case "host":
      return "RAM";
    case "unified":
      return "Unified memory";
    case "storage":
      return "SSD";
  }
}

function LinkEditor({
  link,
  domains,
  networkResources,
  advanced,
  onChange,
}: {
  readonly link: SimLinkSpec;
  readonly domains: readonly MemoryDomainSpec[];
  readonly networkResources: readonly NetworkResourceSpec[];
  readonly advanced: boolean;
  readonly onChange: (update: (link: SimLinkSpec) => SimLinkSpec) => void;
}): React.JSX.Element {
  return (
    <section className="rounded-md border border-zinc-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-bold">{link.id}</h3>
          <div className="truncate text-[11px] text-zinc-500">
            {link.sourceDomainId} → {link.targetDomainId}
          </div>
        </div>
        <Badge variant="neutral">{link.kind}</Badge>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <EditorField label="Source">
          <DomainSelect
            value={link.sourceDomainId}
            domains={domains}
            onChange={(sourceDomainId) => onChange((current) => ({
              ...current,
              sourceDomainId,
            }))}
          />
        </EditorField>
        <EditorField label="Target">
          <DomainSelect
            value={link.targetDomainId}
            domains={domains}
            onChange={(targetDomainId) => onChange((current) => ({
              ...current,
              targetDomainId,
            }))}
          />
        </EditorField>
        <EditorField label="Link kind">
          <Select
            value={link.kind}
            onValueChange={(kind) => onChange((current) => ({
              ...current,
              kind: kind as SimLinkSpec["kind"],
            }))}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {LINK_KINDS.map((kind) => (
                <SelectItem key={kind} value={kind}>{kind}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </EditorField>
        <EditorField label="Bandwidth GB/s">
          <NumberInput
            value={bytesToGigabytesPerSecond(link.bandwidthBytesPerSec)}
            minimum={0.001}
            step={1}
            onChange={(value) => onChange((current) => ({
              ...current,
              bandwidthBytesPerSec: gigabytesPerSecondToBytes(value),
            }))}
          />
        </EditorField>
        <EditorField label="Latency ns">
          <NumberInput
            value={link.latencyNs}
            minimum={0}
            step={1}
            onChange={(latencyNs) => onChange((current) => ({
              ...current,
              latencyNs,
            }))}
          />
        </EditorField>
        <EditorField label="Concurrency lanes">
          <NumberInput
            value={link.concurrencyLanes}
            minimum={1}
            step={1}
            onChange={(concurrencyLanes) => onChange((current) => ({
              ...current,
              concurrencyLanes,
            }))}
          />
        </EditorField>
      </div>
      {advanced && (link.kind === "ethernet" || link.kind === "infiniband")
        ? (
            <div className="mt-4 border-t border-zinc-200 pt-3">
              <div className="mb-2 text-xs font-semibold text-zinc-600">
                Resource path
              </div>
              {networkResources.length === 0
                ? (
                    <p className="text-xs text-zinc-500">
                      No NIC or shared fabric resources are declared.
                    </p>
                  )
                : (
                    <div className="flex flex-wrap gap-2">
                      {networkResources.map((resource) => {
                        const active = link.networkResourceIds?.includes(
                          resource.id,
                        ) ?? false;
                        const order = link.networkResourceIds?.indexOf(
                          resource.id,
                        ) ?? -1;
                        return (
                          <label
                            key={resource.id}
                            className="flex items-center gap-1.5 text-xs text-zinc-700"
                          >
                            <input
                              type="checkbox"
                              checked={active}
                              onChange={() => onChange((current) => ({
                                ...current,
                                networkResourceIds: toggleOrderedValue(
                                  current.networkResourceIds ?? [],
                                  resource.id,
                                ),
                              }))}
                            />
                            {active ? `${order + 1}. ` : ""}
                            {resource.id}
                          </label>
                        );
                      })}
                    </div>
                  )}
            </div>
          )
        : null}
    </section>
  );
}

function SmallLanEditor({
  scenario,
  advanced,
  onChange,
}: {
  readonly scenario: SimulationScenario;
  readonly advanced: boolean;
  readonly onChange: (options: MultiNodeLanOptions) => void;
}): React.JSX.Element {
  const current = smallLanOptions(scenario, advanced);
  const systemCount = new Set(scenario.memoryDomains.map(
    (domain) => domain.nodeId,
  )).size;
  if (systemCount < 2 || systemCount > 4 || current === undefined) {
    return (
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-sm font-bold">Network links</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            Edit directed links individually below.
          </p>
        </div>
      </div>
    );
  }
  const apply = (update: Partial<MultiNodeLanOptions>): void => {
    const nextAdvanced = update.advanced ?? current.advanced ?? false;
    const requestedTransport = update.transport ?? current.transport;
    const transport = !nextAdvanced && requestedTransport === "gpudirect_rdma"
      ? current.linkKind === "infiniband" ? "rdma_host" : "tcp"
      : requestedTransport;
    onChange({ ...current, ...update, advanced: nextAdvanced, transport });
  };
  return (
    <div>
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <h3 className="text-sm font-bold">Small LAN</h3>
          <p className="mt-0.5 text-xs text-zinc-500">
            {systemCount} systems on one switched local network.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-zinc-600">Advanced</span>
          <Switch
            aria-label="Advanced network"
            checked={advanced}
            onCheckedChange={(checked) => apply({ advanced: checked })}
          />
        </div>
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <EditorField label="Network">
          <Select
            value={current.linkKind}
            onValueChange={(kind) => apply({
              linkKind: kind as "ethernet" | "infiniband",
              transport: kind === "infiniband" ? "rdma_host" : "tcp",
            })}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="ethernet">Ethernet</SelectItem>
              <SelectItem value="infiniband">InfiniBand</SelectItem>
            </SelectContent>
          </Select>
        </EditorField>
        <EditorField label="Transport">
          <Select
            value={current.transport}
            onValueChange={(transport) => apply({
              transport: transport as NonNullable<
                MultiNodeLanOptions["transport"]
              >,
            })}
          >
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>
              {NETWORK_TRANSPORTS
                .filter((transport) => advanced || transport !== "gpudirect_rdma")
                .map((transport) => (
                  <SelectItem key={transport} value={transport}>
                    {formatTransport(transport)}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </EditorField>
        <EditorField label="Bandwidth GB/s">
          <NumberInput
            value={bytesToGigabytesPerSecond(
              current.bandwidthBytesPerSec!,
            )}
            minimum={0.001}
            step={1}
            onChange={(value) => apply({
              bandwidthBytesPerSec: gigabytesPerSecondToBytes(value),
            })}
          />
        </EditorField>
        <EditorField label="Latency ns">
          <NumberInput
            value={current.latencyNs!}
            minimum={0}
            step={1}
            onChange={(latencyNs) => apply({ latencyNs })}
          />
        </EditorField>
        <EditorField label="Link lanes">
          <NumberInput
            value={current.linkConcurrencyLanes!}
            minimum={1}
            step={1}
            onChange={(linkConcurrencyLanes) => apply({
              linkConcurrencyLanes,
            })}
          />
        </EditorField>
      </div>
    </div>
  );
}

function smallLanOptions(
  scenario: SimulationScenario,
  advanced: boolean,
): MultiNodeLanOptions | undefined {
  const domainById = new Map(
    scenario.memoryDomains.map((domain) => [domain.id, domain]),
  );
  const link = scenario.links.find((candidate) => {
    if (candidate.kind !== "ethernet" && candidate.kind !== "infiniband") {
      return false;
    }
    const source = domainById.get(candidate.sourceDomainId);
    const target = domainById.get(candidate.targetDomainId);
    return source !== undefined
      && target !== undefined
      && source.nodeId !== target.nodeId;
  });
  if (link === undefined) {
    return undefined;
  }
  const nic = scenario.networkResources?.find(
    (resource) => resource.kind === "nic",
  );
  const fabric = scenario.networkResources?.find(
    (resource) => resource.kind === "switch",
  );
  return {
    advanced,
    linkKind: link.kind as "ethernet" | "infiniband",
    transport: link.transport
      ?? (link.kind === "infiniband" ? "rdma_host" : "tcp"),
    bandwidthBytesPerSec: link.bandwidthBytesPerSec,
    latencyNs: link.latencyNs,
    linkConcurrencyLanes: link.concurrencyLanes,
    nicBandwidthBytesPerSec: nic?.bandwidthBytesPerSec,
    nicLatencyNs: nic?.latencyNs,
    nicConcurrencyLanes: nic?.concurrencyLanes,
    fabricBandwidthBytesPerSec: fabric?.bandwidthBytesPerSec,
    fabricLatencyNs: fabric?.latencyNs,
    fabricConcurrencyLanes: fabric?.concurrencyLanes,
  };
}

function NetworkResourceEditor({
  resources,
  domains,
  onMaterialize,
  onChange,
}: {
  readonly resources: readonly NetworkResourceSpec[];
  readonly domains: readonly MemoryDomainSpec[];
  readonly onMaterialize: () => void;
  readonly onChange: (
    id: string,
    update: (resource: NetworkResourceSpec) => NetworkResourceSpec,
  ) => void;
}): React.JSX.Element {
  if (resources.length === 0) {
    return (
      <section className="border border-dashed border-zinc-300 bg-white p-4">
        <h3 className="text-sm font-bold">Physical network</h3>
        <p className="mt-1 text-xs text-zinc-500">
          This scenario has logical network links but no declared NIC or
          shared fabric resources.
        </p>
        <Button type="button" className="mt-3" onClick={onMaterialize}>
          <Network className="size-4" />
          Add NICs and shared fabric
        </Button>
      </section>
    );
  }
  return (
    <div className="grid gap-3 xl:grid-cols-2">
      {resources.map((resource) => (
        <section
          key={resource.id}
          className="border border-zinc-200 bg-white p-4"
        >
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h3 className="truncate text-sm font-bold">{resource.id}</h3>
              <div className="mt-0.5 text-[11px] text-zinc-500">
                {resource.nodeId ?? "shared across systems"}
              </div>
            </div>
            <Badge variant="neutral">
              {resource.kind === "nic" ? "NIC / HCA" : "switch fabric"}
            </Badge>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-3">
            <EditorField label="Bandwidth GB/s">
              <NumberInput
                value={bytesToGigabytesPerSecond(
                  resource.bandwidthBytesPerSec,
                )}
                minimum={0.001}
                step={1}
                onChange={(value) => onChange(resource.id, (current) => ({
                  ...current,
                  bandwidthBytesPerSec: gigabytesPerSecondToBytes(value),
                }))}
              />
            </EditorField>
            <EditorField label="Latency ns">
              <NumberInput
                value={resource.latencyNs}
                minimum={0}
                step={1}
                onChange={(latencyNs) => onChange(
                  resource.id,
                  (current) => ({ ...current, latencyNs }),
                )}
              />
            </EditorField>
            <EditorField label="Concurrency lanes">
              <NumberInput
                value={resource.concurrencyLanes}
                minimum={1}
                step={1}
                onChange={(concurrencyLanes) => onChange(
                  resource.id,
                  (current) => ({ ...current, concurrencyLanes }),
                )}
              />
            </EditorField>
          </div>
          <div className="mt-4 border-t border-zinc-200 pt-3">
            <div className="mb-2 text-xs font-semibold text-zinc-600">
              Supported transports
            </div>
            <div className="flex flex-wrap gap-2">
              {NETWORK_TRANSPORTS.map((transport) => (
                <label
                  key={transport}
                  className="flex items-center gap-1.5 text-xs text-zinc-700"
                >
                  <input
                    type="checkbox"
                    checked={resource.supportedTransports.includes(transport)}
                    onChange={() => onChange(resource.id, (current) => ({
                      ...current,
                      supportedTransports: toggleOrderedValue(
                        current.supportedTransports,
                        transport,
                      ),
                    }))}
                  />
                  {formatTransport(transport)}
                </label>
              ))}
            </div>
          </div>
          {resource.kind === "nic"
            ? (
                <div className="mt-4 border-t border-zinc-200 pt-3">
                  <div className="mb-2 text-xs font-semibold text-zinc-600">
                    Direct memory access
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {domains
                      .filter((domain) => domain.nodeId === resource.nodeId)
                      .map((domain) => (
                        <label
                          key={domain.id}
                          className="flex items-center gap-1.5 text-xs text-zinc-700"
                        >
                          <input
                            type="checkbox"
                            checked={resource.directMemoryDomainIds.includes(
                              domain.id,
                            )}
                            onChange={() => onChange(
                              resource.id,
                              (current) => ({
                                ...current,
                                directMemoryDomainIds: toggleOrderedValue(
                                  current.directMemoryDomainIds,
                                  domain.id,
                                ),
                              }),
                            )}
                          />
                          {domain.id}
                        </label>
                      ))}
                  </div>
                </div>
              )
            : null}
        </section>
      ))}
    </div>
  );
}

function formatTransport(transport: NetworkTransportMode): string {
  switch (transport) {
    case "tcp":
      return "TCP";
    case "rdma_host":
      return "Host-staged RDMA";
    case "gpudirect_rdma":
      return "GPUDirect RDMA";
  }
}

function toggleOrderedValue<T>(values: readonly T[], value: T): T[] {
  return values.includes(value)
    ? values.filter((entry) => entry !== value)
    : [...values, value];
}

function DomainSelect({
  value,
  domains,
  onChange,
}: {
  readonly value: string;
  readonly domains: readonly MemoryDomainSpec[];
  readonly onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger><SelectValue /></SelectTrigger>
      <SelectContent>
        {domains.map((domain) => (
          <SelectItem key={domain.id} value={domain.id}>
            {domain.id}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function EditorField({
  label,
  className = "",
  children,
}: {
  readonly label: string;
  readonly className?: string;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <label className={`block ${className}`}>
      <span className="mb-1.5 block text-[11px] font-semibold text-zinc-600">
        {label}
      </span>
      {children}
    </label>
  );
}

function TextInput({
  value,
  onChange,
}: {
  readonly value: string;
  readonly onChange: (value: string) => void;
}): React.JSX.Element {
  return (
    <input
      type="text"
      value={value}
      onChange={(event) => onChange(event.currentTarget.value)}
      className="h-9 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:ring-2 focus:ring-sky-500"
    />
  );
}

function NumberInput({
  value,
  minimum,
  maximum,
  step,
  onChange,
}: {
  readonly value: number;
  readonly minimum: number;
  readonly maximum?: number;
  readonly step: number;
  readonly onChange: (value: number) => void;
}): React.JSX.Element {
  return (
    <input
      type="number"
      value={value}
      min={minimum}
      max={maximum}
      step={step}
      onChange={(event) => {
        const next = Number(event.currentTarget.value);
        if (Number.isFinite(next)) {
          onChange(next);
        }
      }}
      className="h-9 w-full rounded-md border border-zinc-300 bg-white px-3 text-sm tabular-nums outline-none focus:ring-2 focus:ring-sky-500"
    />
  );
}

function splitList(value: string): string[] {
  return [...new Set(
    value.split(",").map((entry) => entry.trim()).filter(Boolean),
  )];
}

function toggleCapability(
  current: readonly ComputeCapability[],
  capability: ComputeCapability,
): ComputeCapability[] {
  return current.includes(capability)
    ? current.filter((item) => item !== capability)
    : [...current, capability].sort();
}
