from core.state_machine.second_shoe_states import SecondShoePreparationState
from core.state_machine.state_machine_base import StateMachine
from core.state_machine.lobby_states import LobbyState
from core.state_machine.analysis_states import InitialAnalysisState, InitializeLineState
from core.state_machine.betting_states import FindBetState, WaitBetState, PlaceBetState
from core.state_machine.result_states import (
    WaitResultState, HandleResultState, CheckEndState,
    EndLineState, LeaveTableState, WaitNextGameState, EndSessionState
)
from core.state_machine.context import StateMachineContext

class BaccaratStateMachine(StateMachine):
    def __init__(self, stop_event, is_second_shoe=False, initial_drawdown=None, test_mode=False, strategy="original", minutes_to_run=None):
        self.stop_event = stop_event
        context = StateMachineContext(stop_event, is_second_shoe, initial_drawdown, test_mode, strategy, minutes_to_run)
        initial_state = 'prepare_second_shoe' if is_second_shoe else 'lobby'
        super().__init__(initial_state, context)

    def _create_states(self):
        """Create all state instances with context"""
        return {
            'lobby': LobbyState('lobby', self.context),
            'initial_analysis': InitialAnalysisState('initial_analysis', self.context),
            'initialize_line': InitializeLineState('initialize_line', self.context),
            'prepare_second_shoe': SecondShoePreparationState('prepare_second_shoe', self.context),
            'wait_next_game': WaitNextGameState('wait_next_game', self.context),
            'find_bet': FindBetState('find_bet', self.context),
            'wait_bet': WaitBetState('wait_bet', self.context),
            'place_bet': PlaceBetState('place_bet', self.context),
            'wait_result': WaitResultState('wait_result', self.context),
            'handle_result': HandleResultState('handle_result', self.context),
            'check_end': CheckEndState('check_end', self.context),
            'end_line': EndLineState('end_line', self.context),
            'leave_table': LeaveTableState('leave_table', self.context),
            'end_session': EndSessionState('end_session', self.context)
        }

    def run(self):
        """Main loop that drives the state machine"""
        while not self.stop_event.is_set():
            if not self.update():
                break