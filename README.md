# Customer Value & Acquisition Budget: Management Summary
*Customer Intelligence · 25 Oct 2026·

## Recommendation
Fund **Direct, Email and Organic Search to their caps (€24k)**; they pay back within 90 days even on conservative assumptions. Treat the other **€26k (Affiliate €14k, Paid Search €12k) as conditional**: release it in tranches only after a holdout test confirms the assumed CAC. **Paid Social: €0.**

## 1. Which channels bring valuable customers? (observed data)
| Channel | Avg CAC | 12m margin | 12m margin ÷ CAC | Customers whose 12m margin covers CAC |
|---|---|---|---|---|
| Direct | €6.08 | €64.6 | 10.6x | 80% |
| Email | €9.14 | €83.0 | 9.1x | 83% |
| Organic Search | €12.13 | €75.7 | 6.2x | 80% |
| Affiliate | €31.35 | €64.5 | 2.1x | 55% |
| Paid Search | €48.26 | €67.2 | 1.4x | 47% |
| Paid Social | €42.28 | €53.3 | 1.3x | 44% |

Customers are similarly valuable across channels (next-90d margin €16.6–21.9); **what differs is the price paid to acquire them.** Value is concentrated: the top 10% of customers deliver 45% of next-90d margin, while the 862 customers (21%) with no order in 12m have not repaid their CAC (avg −€31.9).

## 2. Can we predict 90-day margin? (hold-out, n=840)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (mean) | 20.09 | 29.20 | 0.00 |
| Ridge | 17.61 | 26.26 | 0.19 |
| HistGradientBoosting | 17.72 | 26.50 | 0.18 |
| Random Forest (selected on CV) | 17.86 | 26.67 | 0.17 |

The models beat the baseline by about 12% on MAE but are **statistically tied**, so the choice of model does not matter. R² ≈ 0.2 is useful for **ranking**, not for individual forecasts: the top predicted decile averages €51 predicted vs €47 actual and holds 26% of all margin (2.6x lift); the top three deciles hold 52%. Main drivers: trailing 12m margin, days since last order, average order value. No future-period or campaign columns were used as inputs.

## 3. Budget allocation (€50,000, channel caps respected)
| Plan | Incremental customers | Predicted 90d margin | 90d ROI | 12m-proxy ROI |
|---|---|---|---|---|
| **Optimised** (Direct 5k, Email 8k, Organic 11k, Affiliate 14k, Paid Search 12k, Social 0) | 2,905 | €51.6k | **+3.1%** | +348% |
| Equal budget, capped (Direct 5k, Email 8k, others 9.25k) | 2,786 | €49.0k | −2.0% | +323% |
| Only channels ROI-positive at 90d (€24k) | 2,295 | €41.5k | +72.8% | +667% |

Optimising adds about €2.6k margin over an equal split. The first €24k earns +73% within 90 days; the remaining €26k **loses money at 90 days** (Affiliate −55%, Paid Search −68%) and only turns positive on a 12-month view. That is why it is conditional.
*Observed:* each channel's average customer margin. *Assumed (scenario file, not experimentally validated):* incremental CAC and the 90d margin multiplier. Incremental customers = budget ÷ assumed CAC.

## 4. Key risks
- **Assumptions drive the result.** A 25% higher CAC turns optimised ROI to −17.5%; cheap channels (Direct, Organic, Email) likely saturate and include customers who would have bought anyway.
- **Historical value is not lifetime value.** One snapshot, no churn curve, no discounting; customers under 12 months have a truncated 12m window (mature-only ratios are higher: Direct 12.0x, Paid Social 1.4x). The 12m-proxy ROI is an upper-bound sensitivity.
- **Modelling.** 48% of targets are zero, R² ≈ 0.2, only 84 customers per decile (noisy); a single snapshot date means no true out-of-time validation. Re-validate on the next snapshot.
- **Data.** Fixed: one ID collision, one 120% discount, one invalid opt-in value, three missing cells. Unreliable: `days_since_last_order` is populated for all 862 customers with zero orders in 12m.
- **Retention campaign (side finding).** Uplift €5.27 per treated customer (90% CI 3.1–7.3) vs €6.04 cost: net −€0.78. Do not scale as designed; treated customers also had slightly higher prior margin, so the test may not be fully randomised.
