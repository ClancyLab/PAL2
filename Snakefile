rule all:
    input:
        "final_plot.png"

rule visualization:
    input:
        "csv_paths.txt"
    output:
        "final_plot.png"
    script:
        "plots.py"

rule perform_bo:
    input:
        "temp.txt"

rule shift:
    input:
        "output_location.txt"
    output:
        "success.txt",
        temp("temp.txt")
    log:
        "shift_logs.out",
        "shift_logs.err",
    script:
        "scripts/shift_files.py"

rule bayesian_optimization:
    input:
        "user_configuration.yml",
        "default_configuration.json"
    output:
        temp("output_location.txt")
    log:
        "training_logs.log"
    script:
        "scripts/acquisition.py"

# rule predictive_inference:
#     input:
#         "user_configuration.yml",
#         "default_configuration.json"
#     output:
#         "output_location.txt"
#     log:
#         "training_logs.log"
#     script:
#         "scripts/acquisition.py"
        
rule feature_engineering:
    input:
        "user_configuration.yml"
    log:
        "logs/feature_engineering_logs.log"
    script:
        "scripts/feature_selection_methods.py"