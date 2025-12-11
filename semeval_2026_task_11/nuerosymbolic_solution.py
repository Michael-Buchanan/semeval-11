from analysis import (
    LOG_PATH,
    analyze_log_file,
    run_pilot_evaluation,
    run_pilot_evaluation_gpt5_baseline,
    run_neurosymbolic_test_inference,  
)

def main():
    print("=== Syllogistic Reasoning Evaluation ===")
    print("1) Run full pilot evaluation (LLM + solver + write log)")
    print("2) Compute metrics from existing log file")
    print("3) Run full pilot evaluation on baseline LLM (GPT-5)")
    print("4) Run NeuroSymbolic model on test data (Subtask 1 format)") 
    choice = input("Select option (1/2/3/4): ").strip()

    if choice == "1":
        run_pilot_evaluation()
    elif choice == "2":
        path = input(f"Enter log file path (default: {LOG_PATH}): ").strip()
        if not path:
            path = LOG_PATH
        analyze_log_file(path)
    elif choice == "3":
        run_pilot_evaluation_gpt5_baseline()
    elif choice == "4":
        run_neurosymbolic_test_inference()   
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
