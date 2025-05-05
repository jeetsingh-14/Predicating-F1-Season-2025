import os
import sys
import logging
import argparse
import time
from datetime import datetime
import importlib

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create necessary directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/visualizations", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class F1PredictionPipeline:
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()

    def run_script(self, script_name, description):
        """Run a Python script and return True if successful"""
        try:
            script_path = os.path.join("src", script_name)
            if os.path.exists(script_path):
                module_name = script_name.replace(".py", "")
                module = importlib.import_module(f"src.{module_name}")
                if hasattr(module, 'main'):
                    module.main()
                    return True
            logger.error(f"Script {script_path} not found")
            return False
        except Exception as e:
            logger.error(f"Error running {description}: {str(e)}")
            return False

    def run_model_training(self):
        """Run model training and prediction"""
        logger.info("Running Model training and prediction...")
        try:
            # Import and run the model training
            from src.model import main as model_main
            model_main()
            logger.info("Model training and prediction completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            return False

    def run_pipeline(self):
        """Run the complete F1 prediction pipeline"""
        try:
            # Step 1: Update race schedule
            logger.info("Step 1: Updating race schedule...")
            if self.args.update_schedule:
                if not self.run_script("schedule_parser.py", "Schedule parser"):
                    logger.error("Failed to update race schedule. Stopping pipeline.")
                    return False
            
            # Step 2: Analyze news sentiment
            logger.info("Step 2: Analyzing news sentiment...")
            if self.args.include_news:
                if not self.run_script("news_analyzer.py", "News analyzer"):
                    logger.error("Failed to analyze news sentiment. Stopping pipeline.")
                    return False
            
            # Step 3: Fetch betting odds
            logger.info("Step 3: Fetching betting odds...")
            if self.args.include_betting:
                if not self.run_script("betting_odds_analyzer.py", "Betting odds analyzer"):
                    logger.error("Failed to fetch betting odds. Stopping pipeline.")
                    return False
            
            # Step 4: Engineer features
            logger.info("Step 4: Engineering features...")
            if not self.run_script("feature_engineering.py", "Feature engineering"):
                logger.error("Failed to engineer features. Stopping pipeline.")
                return False
            
            # Step 5: Train model and make predictions
            logger.info("Step 5: Training model and making predictions...")
            if not self.run_model_training():
                logger.error("Failed to train model and make predictions. Stopping pipeline.")
                return False
            
            # Step 6: Create visualizations
            logger.info("Step 6: Creating visualizations...")
            if self.args.create_visualizations:
                if not self.run_script("visualization.py", "Visualization"):
                    logger.error("Failed to create visualizations. Stopping pipeline.")
                    return False
            
            # Calculate and log total execution time
            execution_time = time.time() - self.start_time
            logger.info(f"Pipeline completed in {execution_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="F1 Prediction Pipeline")
    
    parser.add_argument(
        "--update-schedule",
        action="store_true",
        help="Update the race schedule before running the pipeline"
    )
    
    parser.add_argument(
        "--include-news",
        action="store_true",
        help="Include news sentiment analysis in the pipeline"
    )
    
    parser.add_argument(
        "--include-betting",
        action="store_true",
        help="Include betting odds analysis in the pipeline"
    )
    
    parser.add_argument(
        "--create-visualizations",
        action="store_true",
        help="Create visualizations after running the pipeline"
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the pipeline
    pipeline = F1PredictionPipeline(args)
    success = pipeline.run_pipeline()
    
    # Exit with appropriate status code
    exit(0 if success else 1)

if __name__ == "__main__":
    main() 