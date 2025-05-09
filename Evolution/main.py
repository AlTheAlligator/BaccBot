import logging
from threading import Event
import argparse
from core.state_machine.baccarat_state_machine import BaccaratStateMachine
from core.modal_monitor import start_modal_monitor_thread
from core.utils import start_keyboard_listener, stop_event
import os

def setup_logging(log_file: str = "log/baccarat_bot.log"):
    """Configures logging to output messages to both console and log file"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler - logs all messages DEBUG and above
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler - logs messages INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Baccarat Bot')
    parser.add_argument('-s', '--second-shoe', action='store_true', help='Run in second shoe mode')
    parser.add_argument('-d', '--drawdown', type=float, help='Current drawdown value (will be converted to negative)')
    parser.add_argument('-t', '--test-mode', action='store_true', help='Run in test mode: place minimal bets every 3rd game only')
    parser.add_argument('--strategy', choices=['original', 'frequency_analysis', 'hybrid_frequency_volatility', 'volatility_adaptive', 'momentum_oscillator', 'chaos_theory', 'neural_oscillator'], default='original',
                      help='Betting strategy to use (default: original)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode: save screenshots to disk for debugging')
    parser.add_argument('-m', '--minutes', type=int, help='Run for specified number of minutes and exit after current line completes')
    args = parser.parse_args()

    if args.second_shoe and args.drawdown is None:
        parser.error("--drawdown is required when using --second-shoe")

    # Convert drawdown to negative if positive
    if args.second_shoe:
        args.drawdown = -abs(args.drawdown)

    # Create required directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    # Setup logging
    setup_logging()
    start_modal_monitor_thread()
    start_keyboard_listener()

    if args.strategy == 'frequency_analysis':
        logging.info("Running with frequency analysis strategy")
    elif args.strategy == 'hybrid_frequency_volatility':
        logging.info("Running with hybrid frequency-volatility strategy")
    elif args.strategy == 'volatility_adaptive':
        logging.info("Running with volatility adaptive strategy")
    elif args.strategy == 'momentum_oscillator':
        logging.info("Running with momentum oscillator strategy")
    elif args.strategy == 'chaos_theory':
        logging.info("Running with chaos theory strategy")
    elif args.strategy == 'neural_oscillator':
        logging.info("Running with neural oscillator strategy")
    else:
        logging.info("Running with original strategy")

    # Set debug mode in screencapture module
    from core.screencapture import set_debug_mode
    set_debug_mode(args.debug)

    if args.debug:
        logging.info("Running in debug mode: screenshots will be saved to disk")

    # Log minutes to run if specified
    if args.minutes:
        logging.info(f"Bot will run for {args.minutes} minutes and exit after current line completes")

    try:
        # Initialize and run state machine with second shoe flag, drawdown, strategy, and minutes to run
        state_machine = BaccaratStateMachine(stop_event, is_second_shoe=args.second_shoe,
                                            initial_drawdown=args.drawdown, test_mode=args.test_mode,
                                            strategy=args.strategy, minutes_to_run=args.minutes)
        state_machine.run()
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, stopping...")
        stop_event.set()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        stop_event.set()
    finally:
        logging.info("Bot stopped.")

if __name__ == "__main__":
    main()
