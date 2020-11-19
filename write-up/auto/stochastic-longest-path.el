(TeX-add-style-hook
 "stochastic-longest-path"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("hyperref" "colorlinks" "urlcolor=blue" "linkcolor=blue" "citecolor=hotpink") ("babel" "british") ("algorithm2e" "linesnumbered" "ruled") ("tcolorbox" "most" "minted")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art12"
    "a4"
    "amsmath"
    "amssymb"
    "xcolor"
    "graphicx"
    "hyperref"
    "booktabs"
    "rotating"
    "caption"
    "babel"
    "algorithm2e"
    "epstopdf"
    "mathtools"
    "subfig"
    "tcolorbox"
    "mdwlist"
    "multirow")
   (TeX-add-symbols
    "R"
    "C"
    "P"
    "E"
    "Var"
    "nbyn"
    "mbyn"
    "l"
    "norm"
    "normi"
    "normo"
    "Chat"
    "e"
    "diff"
    "diag"
    "trace"
    "At"
    "normt"
    "qedsymbol"
    "oldtabcr"
    "nonumberbreak"
    "mynewline"
    "lineref"
    "myvspace"
    "colon")
   (LaTeX-add-labels
    "sect.intro"
    "eq.Li"
    "sect.bounds"
    "subsect.bounds_moments"
    "eq.ui"
    "subsect.bounds_distribution"
    "eq.cdf_convolution"
    "eq.cdf_product"
    "sect.monte_carlo"
    "sect.CLT"
    "subsect.clark_sculli"
    "eq.sum_moments"
    "eq.alpha_beta"
    "eq.clark_max_mu"
    "eq.clark_max_sigma"
    "subsect.correlation_aware"
    "eq.clark_corrs"
    "subsect.canonical"
    "subsect.kamburowski"
    "eq.si_under"
    "eq.si_over"
    "sect.path_reduction"
    "eq.lp_definition"
    "subsect.path_identifying"
    "subsubsect.path_dodin"
    "eq.path_comparison"
    "subsubsect.path_mc"
    "subsect.path_approx"
    "subsect.path_heuristic_bounds"
    "sect.updating"
    "subsect.updating_naive"
    "eq.dash_Li"
    "subsect.updating_backwards"
    "subsect.updating_proposition"
    "sect.results"
    "subsect.graphs"
    "subsubsect.cholesky"
    "tb.cholesky_samples"
    "subsubsect.stg"
    "subsect.empirical_distribution"
    "subsubsect.generating"
    "tb.mc_timings"
    "subsubsect.how_normal"
    "tb.emp_summary"
    "plot.hist_normal"
    "plot.hist_gamma"
    "plot.emp_hists"
    "subsubsect.sensitive"
    "subsect.results_existing"
    "subsubsect.existing_mean"
    "tb.mean_existing"
    "plot.stg_mean"
    "plot.stg_var"
    "plot.stg_mean_var"
    "subsubsect.existing_variance"
    "plot.variance_existing"
    "tb.corlca_direction"
    "subsubsect.existing_efficiency"
    "plot.chol_existing_timings"
    "subsect.results_path"
    "subsect.results_updating"
    "sect.conclusions")
   (LaTeX-add-environments
    "proof"
    "lemma"
    "theorem"
    "prop"
    "code")
   (LaTeX-add-bibliographies
    "references"
    "strings")
   (LaTeX-add-counters
    "mylineno")
   (LaTeX-add-xcolor-definecolors
    "hotpink")
   (LaTeX-add-tcbuselibraries
    "listings"))
 :latex)

