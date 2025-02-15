from core.state_machine.state_machine_base import StateMachine
from core.state_machine.lobby_states import LobbyState
from core.state_machine.analysis_states import InitialAnalysisState, InitializeLineState
from core.state_machine.betting_states import FindBetState, WaitBetState, PlaceBetState
from core.state_machine.result_states import (
    WaitResultState, HandleResultState, CheckEndState,
    EndLineState, LeaveTableState, WaitNextGameState
)
from core.state_machine.context import StateMachineContext

class BaccaratStateMachine(StateMachine):
    def __init__(self, stop_event):
        self.stop_event = stop_event
        context = StateMachineContext(stop_event)
        super().__init__('lobby', context)
        
    def _create_states(self):
        """Create all state instances with context"""
        return {
            'lobby': LobbyState('lobby', self.context),
            'initial_analysis': InitialAnalysisState('initial_analysis', self.context),
            'initialize_line': InitializeLineState('initialize_line', self.context),
            'wait_next_game': WaitNextGameState('wait_next_game', self.context),
            'find_bet': FindBetState('find_bet', self.context),
            'wait_bet': WaitBetState('wait_bet', self.context),
            'place_bet': PlaceBetState('place_bet', self.context),
            'wait_result': WaitResultState('wait_result', self.context),
            'handle_result': HandleResultState('handle_result', self.context),
            'check_end': CheckEndState('check_end', self.context),
            'end_line': EndLineState('end_line', self.context),
            'leave_table': LeaveTableState('leave_table', self.context)
        }
        
    def run(self):
        """Main loop that drives the state machine"""
        while not self.stop_event.is_set():
            if not self.update():
                break