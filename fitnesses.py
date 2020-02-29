class QueensCustom:

    def evaluate(self,state):

        fitness_cnt = 0
        for i in range(len(state) - 1):
            for j in range(i + 1, len(state)):

                if (
                    state[j] != state[i]
                    and state[j] != state[i] + (j - i)
                    and state[j] != state[i] - (j - i)
                ):
                    fitness_cnt += 1

        return fitness_cnt

    def __call__(self, state):
        return self.evaluate(state) # Make it play nice when used as function
