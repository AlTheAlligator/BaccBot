from typing import Optional
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

class State:
    """Base class for all states"""
    def __init__(self, name: str, context):
        self.name = name
        self.context = context
    
    @abstractmethod
    def execute(self) -> Optional[str]:
        """
        Execute the state's logic and return the name of the next state or None.
        None means stay in current state.
        """
        pass

class StateMachine:
    """Simple state machine implementation"""
    def __init__(self, initial_state: str, context):
        self.context = context
        self.states = self._create_states()
        self.current_state = self.states[initial_state]
        logging.info(f"State Machine initialized with state: {initial_state}")
    
    def _create_states(self) -> dict[str, State]:
        """Override this method to create state instances with context"""
        raise NotImplementedError("State machines must implement _create_states")
    
    def update(self) -> bool:
        """
        Update the state machine. Returns False if machine should stop.
        """
        try:
            next_state = self.current_state.execute()
            
            # None means stay in current state
            if next_state is None:
                return True
                
            # Empty string means stop machine
            if next_state == "":
                return False
                
            # Transition to next state
            if next_state in self.states:
                logging.info(f"Transitioning from {self.current_state.name} to {next_state}")
                self.current_state = self.states[next_state]
                return True
            else:
                logging.error(f"Invalid state transition: {next_state}")
                return False
                
        except Exception as e:
            logging.error(f"Error in state {self.current_state.name}: {str(e)}", exc_info=True)
            return False