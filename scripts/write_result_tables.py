"""Generate the research note's numerical content from the completed raw summary."""
import argparse
import json
from pathlib import Path
from statistics import median


def write(root):
    data = json.loads((root/"results/initial-analysis.json").read_text())
    if data["campaign_status"] != "complete" or data["completed_runs"] != 240:
        raise ValueError("The full initial comparison must finish before generating this report")
    docs = root/"docs"
    rows = []
    for item in data["diagnostics"]:
        horizons = {h["steps"]:h for h in item["horizons"]}
        lam = r"$10^{-3}$" if item["regularization"] == 1e-3 else r"$10^{-5}$"
        rows.append(f"{lam} & {item['checkpoints']} & {item['median_score_ratio']:.3f} & "
                    + " & ".join(f"{horizons[h]['median_energy_ratio']:.3f}" for h in (5,10,20))
                    + r" \\")
    (docs/"initial-diagnostic-table.tex").write_text(r"""\begin{center}
\small
\begin{tabular}{lrrrrr}
\hline
$\lambda$ & Checkpoints & $q_{\rm new}/q_{\rm old}$ & $E^{\rm new}_5/E^{\rm old}_5$ & $E^{\rm new}_{10}/E^{\rm old}_{10}$ & $E^{\rm new}_{20}/E^{\rm old}_{20}$ \\
\hline
"""+"\n".join(rows)+r"""
\hline
\end{tabular}
\end{center}
Here $E_h=e_h^{\mathsf T}Ae_h$ after $h$ branch iterations. Entries are medians
of checkpoint ratios; an energy ratio above one favors the old preconditioner.
""")
    labels = {("synthetic","smooth"):"Synthetic (smooth)",
              ("synthetic","random"):"Synthetic (random)",
              ("concrete","observed"):"Concrete", ("airfoil","observed"):"Airfoil"}
    order = [("synthetic","smooth"),("synthetic","random"),("concrete","observed"),("airfoil","observed")]
    workloads = {(w["dataset"],w["target"]):w for w in data["workloads"]}
    rows = []
    for key in order:
        w=workloads[key]
        if w["fully_paired_seeds"] != 5:
            raise ValueError("Report requires five complete converged pairs in each workload")
        times=[w["methods"][m]["median_seconds"] for m in ("cg","fixed","matched_restart","residual_exchange")]
        rows.append(labels[key]+" & "+" & ".join(f"{t:.3f}" for t in times)
                    +f" & {w['median_paired_time_ratio']:.2f}"+r" \\")
    (docs/"initial-timing-table.tex").write_text(r"""\begin{center}
\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\hline
Workload & CG (s) & Fixed (s) & Restarted fixed (s) & Exchange (s) & Time ratio \\
\hline
"""+"\n".join(rows)+r"""
\hline
\end{tabular}
\end{center}
Times are medians across the five seeds of each seed's median over three repeats.
The final column is the median of the five paired exchange/fixed time ratios.
""")
    pairs=data["paired_cases"]
    slower=sum(p["adaptive_over_fixed_time"] > 1 for p in pairs)
    restart_more=sum(median(p["methods"]["matched_restart"]["iterations"])
                     > median(p["methods"]["fixed"]["iterations"]) for p in pairs)
    exchange_more=sum(median(p["methods"]["residual_exchange"]["iterations"])
                      > median(p["methods"]["matched_restart"]["iterations"]) for p in pairs)
    shares=[p["methods"]["residual_exchange"]["median_update_seconds"]/
            p["methods"]["residual_exchange"]["median_time_to_accuracy_seconds"] for p in pairs]
    all_metrics=[m for p in pairs for m in p["methods"].values()]
    residual=max(m["max_true_relative_residual"] for m in all_metrics)
    coeff=max(m["max_relative_solution_error"] for m in all_metrics)
    pred=max(m["max_relative_prediction_difference"] for m in all_metrics)

    def sci(x):
        a,b=f"{x:.2e}".split("e")
        return f"${a}\\times 10^{{{int(b)}}}$"

    text=(f"All {data['completed_runs']} completed timed solves reached the stopping tolerance. "
          f"The exchange method was slower than fixed PCG in {slower} of the {len(pairs)} "
          "paired dataset--target--seed cases. "
          f"Restarted fixed PCG required more iterations than uninterrupted fixed PCG in {restart_more} "
          f"cases; exchanging columns required more iterations than restarted fixed PCG in {exchange_more} cases. "
          f"Exchange-update work accounted for a median {100*median(shares):.1f}\\% of adaptive solve time "
          "across cases. The additional iterations therefore deserve attention alongside update cost.\n\n"
          f"The largest measured true relative residual was {sci(residual)}. The measured maxima of relative coefficient "
          f"error and relative prediction difference from the direct reference were {sci(coeff)} "
          f"and {sci(pred)}, respectively. These maxima cover all methods, seeds, and repeats in this comparison.\n")
    if len(data["source_campaigns"]) > 1:
        text+=("\nThe first execution group reached its 900-second wall-time allowance after 237 completed measurements. "
               "The remaining three measurements were completed separately with the same source, settings, "
               "and prepared data. The interrupted attempt and incomplete record remain preserved.\n")
    (docs/"initial-observations.tex").write_text(text)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path.cwd())
    write(parser.parse_args().root)
