/**
 * Normalisation of the offload placement rules (docs/simulation-model.md §12 "Placement").
 *
 * `OffloadConfig.routedExperts` and `.lookupTables.placements` take one rule or a list of
 * rules over disjoint layer ranges. These helpers give every reader the list form, so a single object and a one-rule
 * list are the same input by construction. Pure; no validation (see `validate.ts`).
 */
import type {
  KvCacheOffloadConfig,
  LookupTableConfig,
  LookupTableRule,
  OffloadConfig,
  RoutedExpertRule,
} from './types';

/**
 * The config's routed-expert rules as a list: absent → `[]`, one object → `[it]`, a list → itself, in the order given
 * (not sorted; the lowering sorts by `layers.from`).
 */
export function expertRules(config: OffloadConfig | undefined): readonly RoutedExpertRule[] {
  return asRuleList(config?.routedExperts);
}

/**
 * The config's lookup-table rules as a list: absent → `[]`, one object → `[it]`, a list → itself, in the order given
 * (not sorted). Rules with tier `'device'` are kept (they place nothing; the resolver skips them).
 */
export function lookupTableRules(config: OffloadConfig | undefined): readonly LookupTableRule[] {
  return asRuleList(lookupTableConfig(config).placements);
}

/** Canonical table policy; old rule/object inputs are interpreted once at this boundary. */
export function lookupTableConfig(config: OffloadConfig | undefined): LookupTableConfig {
  const tables = config?.lookupTables;
  if (tables === undefined) return {};
  if (Array.isArray(tables) || 'tier' in tables) {
    return { placements: tables as LookupTableRule | readonly LookupTableRule[] };
  }
  return tables as LookupTableConfig;
}

/**
 * True if a KV-cache placement moves anything off the device before the unified-memory fold-in: `f_dev < 1`, or a
 * prefix share `f_pre < 1` with a prefix-cache hit (`prefixCacheHitRatio > 0`).
 * See docs/simulation-model.md §12 "KV cache off-device".
 */
export function kvCacheMovesAnything(
  kv: KvCacheOffloadConfig | undefined,
  prefixCacheHitRatio: number | undefined,
): boolean {
  if (kv === undefined) return false;
  const fPre = kv.prefixDeviceFraction ?? kv.deviceFraction;
  return kv.deviceFraction < 1 || ((prefixCacheHitRatio ?? 0) > 0 && fPre < 1);
}

/**
 * Whether placements use an off-device tier (weights, live KV or a cached prefix), before the unified-memory
 * fold-in. A sharing-only table policy needs no host. Remote KV still needs the dedicated device's host attachment.
 * See docs/simulation-model.md §12 "Placement", "KV cache off-device" and "Remote KV pool".
 */
export function hasOffDevicePlacement(
  config: OffloadConfig | undefined,
  prefixCacheHitRatio = 0,
): boolean {
  return (
    expertRules(config).length > 0 ||
    lookupTableRules(config).some((r) => r.tier !== 'device') ||
    (config?.embeddings !== undefined && config.embeddings.tier !== 'device') ||
    kvCacheMovesAnything(config?.kvCache, prefixCacheHitRatio)
  );
}

/** `undefined` → `[]`, a list → itself, one rule → a one-rule list. */
function asRuleList<R extends object>(value: R | readonly R[] | undefined): readonly R[] {
  if (value === undefined) return [];
  // Array.isArray does not narrow readonly arrays; the cast restores the element type.
  return Array.isArray(value) ? (value as readonly R[]) : [value as R];
}
