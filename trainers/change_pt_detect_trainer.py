from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class ChangePtDetectTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ChangePtDetectTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for it in loop:
            loss = self.train_step()
            losses.append(loss)
        
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_step(self):
        batch_x, batch_y, seq_len, y_mask = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.seq_len: seq_len, self.model.y_mask: y_mask}
        _, loss = self.sess.run([self.model.train_step, self.model.loss],
                                feed_dict=feed_dict)
        return loss
