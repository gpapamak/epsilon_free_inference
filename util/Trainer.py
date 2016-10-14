import theano
import theano.tensor as tt
import matplotlib.pyplot as plt

import StepStrategy as ss
import DataStream as ds


isposint = lambda t: isinstance(t, int) and t > 0
dtype = theano.config.floatX


class Trainer:

    def __init__(self, model, trn_data, trn_loss, trn_target=None, val_data=None, val_loss=None, val_target=None, step=ss.Adam()):
        """
        Constructs and configures the trainer.
        :param model: the model to be trained
        :param trn_data: train inputs and (possibly) train targets
        :param trn_loss: theano variable representing the train loss to minimize
        :param trn_target: theano variable representing the train target
        :param val_data: validation inputs and (possibly) validation targets
        :param val_loss: theano variable representing the validation loss
        :param val_target: theano variable representing the validation target
        :param step: step size strategy object
        :return: None
        """

        # parse input
        # TODO: it would be good to type check the other inputs too
        assert isinstance(step, ss.StepStrategy), 'Step must be a step strategy object.'

        # prepare train data
        n_trn_data_list = set([x.shape[0] for x in trn_data])
        assert len(n_trn_data_list) == 1, 'Number of train data is not consistent.'
        self.n_trn_data = list(n_trn_data_list)[0]
        trn_data = [theano.shared(x.astype(dtype)) for x in trn_data]

        # compile theano function for a single training update
        grads = tt.grad(trn_loss, model.parms)
        idx = tt.ivector('idx')
        trn_inputs = [model.input] if trn_target is None else [model.input, trn_target]
        self.make_update = theano.function(
            inputs=[idx],
            outputs=trn_loss,
            givens=zip(trn_inputs, [x[idx] for x in trn_data]),
            updates=step.updates(model.parms, grads)
        )

        # if validation data is given, then set up validation too
        self.do_validation = val_data is not None

        if self.do_validation:

            # prepare validation data
            n_val_data_list = set([x.shape[0] for x in val_data])
            assert len(n_val_data_list) == 1, 'Number of validation data is not consistent.'
            self.n_val_data = list(n_val_data_list)[0]
            val_data = [theano.shared(x.astype(dtype)) for x in val_data]

            # compile theano function for validation
            val_inputs = [model.input] if val_target is None else [model.input, val_target]
            self.validate = theano.function(
                inputs=[],
                outputs=val_loss,
                givens=zip(val_inputs, val_data)
            )

        # initialize some variables
        self.loss = float('inf')
        self.idx_stream = ds.IndexSubSampler(self.n_trn_data)

    def train(self, minibatch=None, tol=None, maxiter=None, verbose=True, monitor_every=None, show_progress=False, val_in_same_plot=True):
        """
        Trains the model.
        :param minibatch: minibatch size
        :param tol: tolerance
        :param maxiter: maximum number of iterations
        :param verbose: if True, print progress during training
        :param monitor_every: monitoring frequency
        :param show_progress: if True, plot training and validation progress
        :param val_in_same_plot: if True, plot validation progress in same plot as training progress
        :return: None
        """

        # parse input
        assert minibatch is None or isposint(minibatch), 'Minibatch size must be a positive integer or None.'
        assert tol is None or tol > 0.0, 'Tolerance must be positive or None.'
        assert maxiter is None or isposint(maxiter), 'Maximum iteration number must be a positive integer or None.'
        assert isinstance(verbose, bool), 'verbose must be boolean.'
        assert monitor_every is None or isposint(monitor_every), 'Monitoring frequency must be a positive integer on None.'
        assert isinstance(show_progress, bool), 'store_progress must be boolean.'
        assert isinstance(val_in_same_plot, bool), 'val_in_same_plot must be boolean.'

        # initialize some variables
        iter = 0
        progress_itr = []
        progress_trn = []
        progress_val = []
        minibatch = self.n_trn_data if minibatch is None else minibatch
        maxiter = float('inf') if maxiter is None else maxiter
        monitor_every = float('inf') if monitor_every is None else monitor_every

        # main training loop
        while True:

            # make update to parameters
            trn_loss = self.make_update(self.idx_stream.gen(minibatch))
            diff = self.loss - trn_loss
            iter += 1
            self.loss = trn_loss

            if iter % monitor_every == 0:

                # do validation
                if self.do_validation:
                    val_loss = self.validate()

                # monitor progress
                if show_progress:
                    progress_itr.append(iter)
                    progress_trn.append(trn_loss)
                    if self.do_validation: progress_val.append(val_loss)

                # print info
                if verbose:
                    if self.do_validation:
                        print 'Iteration = {0}, train loss = {1}, validation loss = {2}'.format(iter, trn_loss, val_loss)
                    else:
                        print 'Iteration = {0}, train loss = {1}'.format(iter, trn_loss)

            # check for convergence
            if abs(diff) < tol or iter >= maxiter:
                break

        # plot progress
        if show_progress:

            if self.do_validation:

                if val_in_same_plot:
                    fig, ax = plt.subplots(1, 1)
                    ax.semilogx(progress_itr, progress_trn, 'b', label='training')
                    ax.semilogx(progress_itr, progress_val, 'r', label='validation')
                    ax.set_xlabel('iterations')
                    ax.set_ylabel('loss')
                    ax.legend()

                else:
                    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    ax1.semilogx(progress_itr, progress_trn, 'b')
                    ax2.semilogx(progress_itr, progress_val, 'r')
                    ax2.set_xlabel('iterations')
                    ax1.set_ylabel('training loss')
                    ax2.set_ylabel('validation loss')

            else:
                fig, ax = plt.subplots(1, 1)
                ax.semilogx(progress_itr, progress_trn, 'b')
                ax.set_xlabel('iterations')
                ax.set_ylabel('training loss')
                ax.legend()

            plt.show(block=False)
