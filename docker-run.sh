#!/bin/bash
# Script to run the Obesity ML Project with Docker
# Makes it easy to run different services without remembering docker commands

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║            OBESITY ML PROJECT - DOCKER HELPER                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

USAGE:
    ./docker-run.sh [COMMAND]

COMMANDS:

  PIPELINE COMMANDS:
    eda           Run the EDA pipeline (data cleaning)
    visualize     Generate EDA visualizations (PNG images)
    compare       Compare datasets (validate results)
    test          Run unit tests
    all           Run EDA + Compare + Test (complete workflow)

  SERVER COMMANDS:
    mlflow        Start MLflow UI (http://localhost:5000)
    shell         Open interactive bash shell inside container

  MANAGEMENT COMMANDS:
    build         Build Docker images
    clean         Remove all containers and images
    logs          Show logs from running containers
    stop          Stop all running containers

  ℹHELP:
    help          Show this help message

EXAMPLES:

  # Run complete workflow (recommended for first time)
  ./docker-run.sh all

  # Run only EDA pipeline
  ./docker-run.sh eda

  # Generate visualizations
  ./docker-run.sh visualize

  # Compare results
  ./docker-run.sh compare

  # Run tests
  ./docker-run.sh test

  # Start MLflow UI and keep it running
  ./docker-run.sh mlflow

  # Open shell for exploration
  ./docker-run.sh shell

  # Clean everything and start fresh
  ./docker-run.sh clean
  ./docker-run.sh build
  ./docker-run.sh all

EOF
}

# Function to build images
build_images() {
    print_message "$BLUE" "Building Docker images..."
    docker-compose build
    print_message "$GREEN" "Build complete!"
}

# Function to run EDA pipeline
run_eda() {
    print_message "$BLUE" "Running EDA pipeline..."
    docker-compose run --rm eda-pipeline
    print_message "$GREEN" "EDA pipeline complete!"
}

# Function to generate visualizations
run_visualize() {
    print_message "$BLUE" "Generating EDA visualizations..."
    docker-compose run --rm visualize
    print_message "$GREEN" "Visualizations generated in reports/figures/"
}

# Function to compare datasets
run_compare() {
    print_message "$BLUE" "Comparing datasets..."
    docker-compose run --rm compare
}

# Function to run tests
run_tests() {
    print_message "$BLUE" "Running tests..."
    docker-compose run --rm test
}

# Function to run complete workflow
run_all() {
    print_message "$YELLOW" "Running complete workflow..."
    echo ""
    
    print_message "$BLUE" "Step 1/4: Running EDA pipeline..."
    run_eda
    echo ""
    
    print_message "$BLUE" "Step 2/4: Generating visualizations..."
    run_visualize
    echo ""
    
    print_message "$BLUE" "Step 3/4: Comparing datasets..."
    run_compare
    echo ""
    
    print_message "$BLUE" "Step 4/4: Running tests..."
    run_tests
    echo ""
    
    print_message "$GREEN" "╔════════════════════════════════════════════════════════════╗"
    print_message "$GREEN" "║                                                            ║"
    print_message "$GREEN" "║              COMPLETE WORKFLOW FINISHED!                   ║"
    print_message "$GREEN" "║                                                            ║"
    print_message "$GREEN" "╚════════════════════════════════════════════════════════════╝"
}

# Function to start MLflow UI
start_mlflow() {
    print_message "$BLUE" "Starting MLflow UI..."
    print_message "$YELLOW" "Access MLflow at: http://localhost:5000"
    print_message "$YELLOW" "Press Ctrl+C to stop"
    docker-compose up mlflow
}

# Function to open shell
open_shell() {
    print_message "$BLUE" "Opening interactive shell..."
    print_message "$YELLOW" "Type 'exit' to leave the shell"
    docker-compose run --rm shell bash
}

# Function to show logs
show_logs() {
    print_message "$BLUE" "Showing logs..."
    docker-compose logs --tail=100 -f
}

# Function to stop containers
stop_containers() {
    print_message "$BLUE" "Stopping all containers..."
    docker-compose down
    print_message "$GREEN" "All containers stopped!"
}

# Function to clean everything
clean_all() {
    print_message "$YELLOW" "Cleaning all containers and images..."
    docker-compose down --rmi all --volumes --remove-orphans
    print_message "$GREEN" "Cleanup complete!"
}

# Main script logic
case "${1:-help}" in
    eda)
        run_eda
        ;;
    visualize)
        run_visualize
        ;;
    compare)
        run_compare
        ;;
    test)
        run_tests
        ;;
    all)
        run_all
        ;;
    mlflow)
        start_mlflow
        ;;
    shell)
        open_shell
        ;;
    build)
        build_images
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_containers
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_message "$RED" "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
