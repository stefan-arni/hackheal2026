/**
 * The boundary between "computed" and "shown to a clinician".
 *
 * Acceptance rule from the authored battery: reject display of any `reportable: false` metric.
 * Whether a value may reach a clinician is a clinical decision recorded per metric on the card, so
 * this filter is driven by the CARD, never by code defaults, and it is deny-by-default in both
 * directions: a metric the card marks unreportable is dropped, and so is a metric the card never
 * declared at all, because a value nobody approved has no business on a clinical surface either.
 */

interface DeclaredMetric {
  readonly id?: unknown;
  readonly reportable?: unknown;
}

/** Returns only the values whose metric id the card declares with `reportable: true`. */
export function filterReportable(
  card: Readonly<Record<string, unknown>>,
  values: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const declared = card.derived_metrics;
  const allowed = new Set<string>();
  if (Array.isArray(declared)) {
    for (const metric of declared as DeclaredMetric[]) {
      if (typeof metric.id === "string" && metric.reportable === true) {
        allowed.add(metric.id);
      }
    }
  }
  const shown: Record<string, unknown> = {};
  for (const [id, value] of Object.entries(values)) {
    if (allowed.has(id)) shown[id] = value;
  }
  return shown;
}
