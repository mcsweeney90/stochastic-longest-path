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
    "mdwlist")
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
    "sect.normality"
    "eq.sum_moments"
    "eq.alpha_beta"
    "eq.clark_max_mu"
    "eq.clark_max_sigma"
    "subsect.correlation_aware"
    "subsect.kamburowski"
    "eq.si_under"
    "eq.si_over"
    "sect.other_methods"
    "sect.updating"
    "subsect.remaining"
    "subsect.corr_update"
    "subsect.propagating"
    "eq.dash_Li"
    "sect.results"
    "subsect.before_runtime"
    "subsubsect.results_normality"
    "plot.normal_assum_normal"
    "subsubsect.results_bounds"
    "subsubsect.results_correlations"
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

