#!/bin/bash

# Function to check if a command was successful
check_command() {
    if [ $? -ne 0 ]; then
        echo "Error: $1"
        exit 1
    fi
}

# Add GPU IDs to the global scope
GPU_IDS=""

# Phase 0
phase0() {
    echo "Starting Phase 0..."

    # Delete RallyClipper/outputs if it exists
    if [ -d "RallyClipper/outputs" ]; then
        rm -rf RallyClipper/outputs
        check_command "Failed to delete RallyClipper/outputs"
    fi

    # Prepare BallTracker/matches folder
    if [ -d "BallTracker/matches" ]; then
        rm -rf BallTracker/matches
    fi
    mkdir -p BallTracker/matches
    check_command "Failed to create BallTracker/matches"

    # Prepare BallTracker/outputs folder
    if [ -d "BallTracker/outputs" ]; then
        rm -rf BallTracker/outputs
    fi
    mkdir -p BallTracker/outputs
    check_command "Failed to create BallTracker/outputs"

    # Prepare TableTracker/outputs folder
    if [ -d "TableTracker/outputs" ]; then
        rm -rf TableTracker/outputs
    fi
    mkdir -p TableTracker/outputs
    check_command "Failed to create TableTracker/outputs"

    # Prepare PaddleTracker/outputs folder
    if [ -d "PaddleTracker/outputs" ]; then
        rm -rf PaddleTracker/outputs
    fi
    mkdir -p PaddleTracker/outputs
    check_command "Failed to create PaddleTracker/outputs"

    # Prepare HumanPoseTracker/outputs folder
    if [ -d "HumanPoseTracker/outputs" ]; then
        rm -rf HumanPoseTracker/outputs
    fi
    mkdir -p HumanPoseTracker/outputs
    check_command "Failed to create HumanPoseTracker/outputs"
}

# Modified phase1 function
phase1() {
    echo "Starting Phase 1..."
    
    # Activate conda environment
    conda activate general
    check_command "Failed to activate general environment"

    # Run inference with GPU IDs
    cd RallyClipper
    python main.py matches outputs --gpu_ids $GPU_IDS
    check_command "Failed to run RallyClipper/main.py"
    cd ..

    # Deactivate conda environment
    conda deactivate
    echo "Phase 1 completed successfully."
}

# Modified phase2 function
phase2() {
    echo "Starting Ball Tracking..."

    # Copy outputs
    cp -R RallyClipper/outputs/* BallTracker/matches/
    check_command "Failed to copy outputs to BallTracker/matches"

    # Activate conda environment
    conda activate ball
    check_command "Failed to activate ball environment"

    # Run commands with GPU IDs
    cd BallTracker
    python main.py matches --gpu_ids $GPU_IDS
    check_command "Failed to run main.py in BallTracker"

    python clip.py matches outputs
    check_command "Failed to run clip.py in BallTracker"
    cd ..

    # Deactivate conda environment
    conda deactivate
    echo "Phase 2 completed successfully."
}

# Modified phase3 function
phase3() {
    echo "Starting Table Tracking..."

    # Activate conda environment
    conda activate yolo
    check_command "Failed to activate table environment"

    # Run command with GPU IDs
    cd TableTracker
    python main.py ../BallTracker/matches --gpu_ids $GPU_IDS
    check_command "Failed to run main.py in TableTracker"
    cd ..

    # Deactivate conda environment
    conda deactivate
    echo "Phase 3 completed successfully."
}

# Modified phase4 function
phase4() {
    echo "Starting Paddle Tracking..."

    # Activate conda environment
    conda activate yolo
    check_command "Failed to activate paddle environment"

    # Run command with GPU IDs
    cd PaddleTracker
    python main.py ../BallTracker/matches --gpu_ids $GPU_IDS
    check_command "Failed to run main.py in PaddleTracker"
    cd ..

    # Deactivate conda environment
    conda deactivate
    echo "Phase 4 completed successfully."
}

# Modified phase5 function
phase5() {
    echo "Starting Phase 5..."

    # Activate conda environment
    conda activate pose
    check_command "Failed to activate pose environment"

    # Run command with GPU IDs
    cd HumanPoseTracker
    python main.py ../BallTracker/matches --gpu_ids $GPU_IDS
    check_command "Failed to run main.py in HumanPoseTracker"
    cd ..

    # Deactivate conda environment
    conda deactivate
    echo "Phase 5 completed successfully."
}

# Phase 6
phase6() {
    echo "Starting Phase 6..."

    # Create or clear the final outputs folder
    if [ -d "outputs" ]; then
        rm -rf outputs
    fi
    mkdir -p outputs
    check_command "Failed to create outputs folder"

    # Process each match folder
    for match_folder in HumanPoseTracker/outputs/*/; do
        match_name=$(basename "$match_folder")
        echo "Processing match: $match_name"

        # Create match folder in the final outputs
        mkdir -p "outputs/$match_name"

        # Process each name folder within the match
        for name_folder in "$match_folder"*/; do
            name=$(basename "$name_folder")
            echo "  Processing name: $name"

            # Create name folder in the final outputs
            mkdir -p "outputs/$match_name/$name"

            # Copy files from HumanPoseTracker/outputs
            pose_files="$name_folder"*
            if ls $pose_files 1> /dev/null 2>&1; then
                cp -R $pose_files "outputs/$match_name/$name/"
                check_command "Failed to copy HumanPoseTracker files for $match_name/$name"
            else
                echo "    Warning: HumanPoseTracker files not found for $match_name/$name"
            fi

            # Copy table CSV if it exists
            table_csv="TableTracker/outputs/$match_name/${name}_table.csv"
            if [ -f "$table_csv" ]; then
                cp "$table_csv" "outputs/$match_name/$name/"
                check_command "Failed to copy table CSV for $match_name/$name"
            else
                echo "    Warning: TableTracker CSV not found for $match_name/$name"
            fi

            # Copy paddle CSV if it exists
            paddle_csv="PaddleTracker/outputs/$match_name/${name}_paddle.csv"
            if [ -f "$paddle_csv" ]; then
                cp "$paddle_csv" "outputs/$match_name/$name/"
                check_command "Failed to copy paddle CSV for $match_name/$name"
            else
                echo "    Warning: PaddleTracker CSV not found for $match_name/$name"
            fi

            # Copy ball CSV if it exists
            ball_csv="BallTracker/outputs/$match_name/${name}_ball.csv"
            if [ -f "$ball_csv" ]; then
                cp "$ball_csv" "outputs/$match_name/$name/"
                check_command "Failed to copy ball CSV for $match_name/$name"
            else
                echo "    Warning: Ball CSV not found for $match_name/$name"
            fi

            # Copy mp4 if it exists
            mp4="BallTracker/matches/$match_name/${name}.mp4"
            if [ -f "$mp4" ]; then
                cp "$mp4" "outputs/$match_name/$name/"
                check_command "Failed to copy mp4 for $match_name/$name"
            else
                echo "    Warning: mp4 not found for $match_name/$name"
            fi
        done
    done

    # Create the zip file with timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    zip_file="outputs_${timestamp}.zip"
    zip -r "$zip_file" "outputs"
    check_command "Failed to create zip file"

    echo "Phase 6 completed successfully. Final outputs organized in 'outputs' folder."
}  

# Modified phase7 function
phase7() {
    echo "Starting Phase 7..."

    # Activate conda environment
    conda activate general
    check_command "Failed to activate general environment"

    # Run command with GPU IDs
    cd RallyReconstructor
    python main.py ../outputs
    check_command "Failed to run main.py in RallyReconstructor"
    cd ..

    # Create the zip file with timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    zip_file="recons_${timestamp}.zip"
    zip -r "$zip_file" "RallyReconstructor/recons"
    check_command "Failed to create zip file"

    echo "Phase 7 completed successfully."
}

# Check if matches folder exists in RallyClipper
check_RallyClipper_matches() {
    if [ ! -d "RallyClipper/matches" ]; then
        echo "Error: RallyClipper/matches folder not found"
        exit 1
    fi
}

# Modified main function to accept GPU IDs
main() {
    eval "$(conda shell.bash hook)"

    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu_ids)
                GPU_IDS="$2"
                shift 2
                ;;
            *)
                POSITIONAL_ARGS+=("$1")
                shift
                ;;
        esac
    done

    # Check if GPU IDs are provided
    if [ -z "$GPU_IDS" ]; then
        echo "Error: GPU IDs not provided. Use --gpu_ids flag."
        exit 1
    fi

    case "${POSITIONAL_ARGS[0]}" in
        phase0)
            phase0
            ;;
        phase1)
            check_RallyClipper_matches
            phase1
            ;;
        phase2)
            phase2
            ;;
        phase3)
            phase3
            ;;
        phase4)
            phase4
            ;;
        phase5)
            phase5
            ;;
        phase6)
            phase6
            ;;
        phase7)
            phase7
            ;;
        all)
            check_RallyClipper_matches
            phase0
            phase1
            phase2
            phase3
            phase4
            phase5
            phase6
            phase7
            ;;
        *)
            echo "Usage: $0 --gpu_ids 'GPU_IDS' {phase1|phase2|phase3|phase4|phase5|phase6|all}"
            exit 1
            ;;
    esac
}

# Run the main function with command-line arguments
main "$@"