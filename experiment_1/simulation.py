"""
WIP
"""
class Simulation():

    def __init__(self,
                 model: NenwinModel,
                 input_placer: InputPlacer,
                 output_reader: OutputReader,
                 pipe_end: multiprocessing.connection.Connection)
        self.__model = model
        self.__input_placer = input_placer
        self.__pipe = pipe_end

    def run(max_num_steps: Number = float("inf")):
        pass

    def __handle_commands(self):
        """
        Reads command given through the pipe and
        executes it. 
        Does nothing if no command exists in the queue.
        """
        if self.__pipe_end.poll():
            message = self.__pipe_end.recv()
            assert isinstance(message, UIMessage)

            command = message.command
            if command == UICommands.stop:
                assert False, "TODO"
            elif command == UICommands.write_output:
                self._produce_outputs()
            elif command == UICommands.read_input:
                self._handle_inputs(message.data)

    # def run(self, max_num_steps: Number = float("inf")):
    #     """
    #     Start simulation and imput processing until stop signal is received.
    #     While running, will accept inputs, and produce outputs when requested.

    #     By default runs indefinitely until stop signal is received.
    #     An optional max amount of timesteps can be given
    #     (convenient for testing).
    #     """
    #     num_remaining_steps = max_num_steps

    #     while num_remaining_steps > 0:
    #         num_remaining_steps -= 1

    #         self.__handle_commands()

    #         