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
    "eq.ui"
    "eq.cdf_convolution"
    "eq.cdf_product"
    "sect.heuristics"
    "subsect.monte_carlo"
    "subsect.normality"
    "subsubsect.clark_sculli"
    "eq.sum_moments"
    "eq.alpha_beta"
    "eq.clark_max_mu"
    "eq.clark_max_sigma"
    "subsubsect.correlation_aware"
    "eq.clark_corrs"
    "subsubsect.canonical"
    "subsubsect.kamburowski"
    "eq.si_under"
    "eq.si_over"
    "sect.path_heuristic"
    "eq.lp_definition"
    "subsect.identifying"
    "eq.path_comparison"
    "subsect.approx_max"
    "subsect.discussion"
    "sect.updating"
    "subsect.remaining"
    "subsect.corr_update"
    "subsect.propagating"
    "eq.dash_Li"
    "sect.results"
    "subsect.graphs"
    "subsubsect.cholesky"
    "tb.cholesky_samples"
    "subsubsect.stg"
    "subsect.empirical_distribution"
    "subsubsect.how_normal"
    "tb.emp_summary"
    "plot.hist_normal"
    "plot.hist_gamma"
    "plot.emp_hists"
    "subsubsect.samples"
    "tb.mc_timings"
    "plot.mc_variance"
    "subsect.results_existing"
    "tb.mean_existing"
    "plot.variance_existing"
    "tb.direction_var"
    "plot.existing_timings"
    "subsect.results_path_reduction"
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

