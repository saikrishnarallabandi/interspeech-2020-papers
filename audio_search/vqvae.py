import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dsp import *
import sys
import time
from overtone import Overtone
from vector_quant import VectorQuant
from downsampling_encoder import DownsamplingEncoder
from query_encoder import QueryEncoder
from lstm_speaker import LSTMSpeakerIdentifier
import env as env
import logger as logger
import random



class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        batch_size, time_steps = x.size(0), x.size(1)
        x = x.contiguous()
        x = x.view(batch_size * time_steps, -1)
        x = self.module(x)
        x = x.contiguous()
        x = x.view(batch_size, time_steps, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, upsample_factors, normalize_vq=False,
            noise_x=False, noise_y=False):
        super().__init__()
        self.n_classes = 256
        self.overtone = Overtone(rnn_dims, fc_dims, 128, 256)
        self.vq = VectorQuant(1, 512, 128, normalize=normalize_vq)
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.discrete_dir_path = "/home/srallaba/projects/text2speech/repos/Kinjal/WaveRNN/arctic_data/search_discrete/"

        encoder_layers = [
            (2, 4, 1),
            (2, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
            (2, 4, 1),
            (1, 4, 1),
        ]
        self.encoder = DownsamplingEncoder(128, encoder_layers)
        # self.speaker_lstm = LSTMSpeakerIdentifier()
        self.frame_advantage = 15
        self.num_params()
        self.searchNquery2label = SequenceWise(nn.Linear(128, 1))

    def forward(self, mel_input, samples_input, mel_query, samples_query):
        # x: (N, 768, 3)
        # logger.log(f'x: {x.size()}')
        # samples: (N, 1022)
        # logger.log(f'samples: {samples.size()}')
        continuous_input = self.encoder(samples_input)
        continuous_query = self.encoder(samples_query)
        # continuous: (N, 14, 64)
        # logger.log(f'continuous: {continuous.size()}')
        discrete_input, vq_pen_input, encoder_pen_input, entropy_input = self.vq(continuous_input.unsqueeze(2))
        discrete_query, vq_pen_query, encoder_pen_query, entropy_query = self.vq(continuous_query.unsqueeze(2))
        # discrete: (N, 14, 1, 64)
        # logger.log(f'discrete: {discrete.size()}')

        # cond: (N, 768, 64)
        # logger.log(f'cond: {cond.size()}')
        speaker_id_input = self.speaker_lstm.forward(mel_input)
        speaker_id_query = self.speaker_lstm.forward(mel_query)

        coarse_search, fine_search = self.overtone(search_quant, discrete_search.squeeze(2), speaker_id_search)
        # // something for discrete_query and discrete_input to predict 0/1
        #     // downsize the search to query dimension for dot product using cnn
        prediction = []
        for discrete_search_, discrete_query_ in zip(discrete_search, discrete_query):
            num_pad = discrete_search_.shape[0] - discrete_query_.shape[0]
            discrete_query_ = torch.LongTensor([np.concatenate([discrete_query_, np.zeros(num_pad, dtype=np.int16)])])
            discrete_query_.transpose(0, 1)
            prediction.append(torch.dot(discrete_query_, discrete_search_))
        return coarse_search, fine_search, vq_pen_search.mean(), encoder_pen_search.mean(), entropy_search, prediction
        return self.overtone(x, discrete.squeeze(2)), vq_pen.mean(), encoder_pen.mean(), entropy

    def after_update(self):
        self.overtone.after_update()
        self.vq.after_update()

    def forward_generate(self, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False):
        if use_half:
            samples = samples.half()
        # samples: (L)
        #logger.log(f'samples: {samples.size()}')
        self.eval()
        with torch.no_grad() :
            continuous = self.encoder(samples)
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
            #logger.log(f'entropy: {entropy}')
            # cond: (1, L1, 64)
            #logger.log(f'cond: {cond.size()}')
            output = self.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half, verbose=verbose)
        self.train()
        return output

    def forward_QbE(self, search_samples, encoded_query_mfcc, pooling_model, index, deterministic=False, use_half=False, verbose=False):
        #self.eval()
        self.encoder.eval()
        self.vq.eval()
        continuous_search = []
        #for search_sample, encoded_query_ in zip(search_samples, encoded_query_mfcc):
        #    print("Shapes input to the model: ", search_sample.shape, encoded_query_.shape)
        continuous_sample_search = self.encoder(search_samples)
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous_sample_search.unsqueeze(2))
            #print(discrete.shape)
        #print("Shape of discrete and encoded_query_mfcc: ", discrete.shape, encoded_query_mfcc.shape)

        discrete = discrete.squeeze(2) #.squeeze(0)
        #print("Shape of discrete and encoded_query_mfcc: ", discrete.shape, encoded_query_mfcc.shape)

        encoded_query_mfcc = encoded_query_mfcc.unsqueeze_(1).expand(encoded_query_mfcc.shape[0], discrete.shape[1], encoded_query_mfcc.shape[2])
            #print("Shape of discrete and encoded_query_: ", discrete.shape, encoded_query_.shape)

            #np.save(self.discrete_dir_path + str(i), discrete.cpu().detach().numpy())
            #num_pad = encoded_query_.shape[0] - discrete.shape[0]
            #if num_pad > 0:
            #    encoded_query_ = encoded_query_[:discrete.shape[0], :]
            #else:
            #    discrete = discrete[:encoded_query_.shape[0], :]

            #discrete = pooling_model(discrete.unsqueeze(1)).squeeze(1).squeeze(1)
            #encoded_query_ = pooling_model(encoded_query_.unsqueeze(1)).squeeze(1).squeeze(1)
        prediction = torch.tanh(self.searchNquery2label(encoded_query_mfcc + discrete))
        prediction, _ = torch.max(prediction, dim=1)
        #print("Shape of prediction: ", prediction.shape)
        return prediction.squeeze(1)


    def forward_QbE_orig(self, search_samples, encoded_query_mfcc, pooling_model, index, deterministic=False, use_half=False, verbose=False):
        #self.eval()
        self.encoder.eval()
        self.vq.eval()
        continuous_search = []
        prediction = []
        i = index
        print(i)
        for search_sample, encoded_query_ in zip(search_samples, encoded_query_mfcc):
            continuous_sample_search = self.encoder(search_sample.unsqueeze(0))
            discrete, vq_pen, encoder_pen, entropy = self.vq(continuous_sample_search.unsqueeze(2))
            #print(discrete.shape)
            discrete = discrete.squeeze(2).squeeze(0)
            print(type(discrete))
            print(type(i))
            np.save(self.discrete_dir_path + str(i), discrete.cpu().detach().numpy())
            num_pad = encoded_query_.shape[0] - discrete.shape[0]
            if num_pad > 0:
                encoded_query_ = encoded_query_[:discrete.shape[0], :]
            else:
                discrete = discrete[:encoded_query_.shape[0], :]

            discrete = pooling_model(discrete.unsqueeze(1)).squeeze(1).squeeze(1)
            encoded_query_ = pooling_model(encoded_query_.unsqueeze(1)).squeeze(1).squeeze(1)
            print("Shape of discrete and encoded_query_: ", discrete.shape, encoded_query_.shape)
            prediction.append(F.sigmoid(torch.dot(discrete, encoded_query_)))
            i += 1

        prediction = torch.stack(prediction, dim=0)
        return prediction

    def compute_loss(self, prediction, actual):
        return nn.BCELoss(nn.Sigmoid().forward(prediction), actual)

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        logger.log('Trainable Parameters: %.3f million' % parameters)

    def load_state_dict(self, dict, strict=False):
        if strict:
            return super().load_state_dict(self.upgrade_state_dict(dict))
        else:
            my_dict = self.state_dict()
            new_dict = {}
            for key, val in dict.items():
                if key not in my_dict:
                    logger.log(f'Ignoring {key} because no such parameter exists')
                elif val.size() != my_dict[key].size():
                    logger.log(f'Ignoring {key} because of size mismatch')
                else:
                    logger.log(f'Loading {key}')
                    new_dict[key] = val
            return super().load_state_dict(new_dict, strict=False)

    def upgrade_state_dict(self, state_dict):
        out_dict = state_dict.copy()
        return out_dict

    def freeze_encoder(self):
        for name, param in self.named_parameters():
            if name.startswith('encoder.') or name.startswith('vq.'):
                logger.log(f'Freezing {name}')
                param.requires_grad = False
            else:
                logger.log(f'Not freezing {name}')

    def pad_left(self):
        return max(self.pad_left_decoder(), self.pad_left_encoder())

    def pad_left_decoder(self):
        return self.overtone.pad()

    def pad_left_encoder(self):
        return self.encoder.pad_left + (
                self.overtone.cond_pad - self.frame_advantage) * self.encoder.total_scale

    def pad_right(self):
        return self.frame_advantage * self.encoder.total_scale

    def total_scale(self):
        return self.encoder.total_scale

    def do_train(self, paths, dataset, optimiser, epochs, batch_size, step, lr=1e-4, valid_index=[],
                 use_half=False, do_clip=False):

        if use_half:
            import apex
            optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
        for p in optimiser.param_groups: p['lr'] = lr
        criterion = nn.NLLLoss().cuda()
        k = 0
        saved_k = 0
        pad_left = self.pad_left()
        pad_left_encoder = self.pad_left_encoder()
        pad_left_decoder = self.pad_left_decoder()
        if self.noise_x:
            extra_pad_right = 127
        else:
            extra_pad_right = 0
        pad_right = self.pad_right() + extra_pad_right
        window = 16 * self.total_scale()
        logger.log(
            f'pad_left={pad_left_encoder}|{pad_left_decoder}, pad_right={pad_right}, total_scale={self.total_scale()}')

        for e in range(epochs):
            trn_loader = DataLoader(dataset,
                                    collate_fn=lambda batch: env.collate_samples(pad_left, window, pad_right, batch),
                                    batch_size=16,
                                    num_workers=0, shuffle=True, pin_memory=True)

            start = time.time()
            running_loss_c = 0.
            running_loss_f = 0.
            running_loss_vq = 0.
            running_loss_vqc = 0.
            running_entropy = 0.
            running_max_grad = 0.
            running_loss_ce_label = 0.
            running_max_grad_name = ""

            iters = len(trn_loader)

            # enumerate mfcc, mel, quant for search, mfcc for query, and label
            # search_wave16 : quant
            for i, (search_wave16, search_mel16, query_mfcc16, label) in enumerate(trn_loader):
                search_wave16 = search_wave16.cuda()
                search_mel16 = search_mel16.cuda()
                query_mfcc16 = query_mfcc16.cuda()
                label = label.cuda()
                coarse = (search_wave16 + 2 ** 15) // 256
                fine = (search_wave16 + 2 ** 15) % 256
                coarse_f = coarse.float() / 127.5 - 1.
                fine_f = fine.float() / 127.5 - 1.
                total_f = (search_wave16.float() + 0.5) / 32767.5

                if self.noise_y:
                    noisy_f = total_f * (
                                0.02 * torch.randn(total_f.size(0), 1).cuda()).exp() + 0.003 * torch.randn_like(total_f)
                else:
                    noisy_f = total_f

                if use_half:
                    coarse_f = coarse_f.half()
                    fine_f = fine_f.half()
                    noisy_f = noisy_f.half()

                x = torch.cat([
                    coarse_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                    fine_f[:, pad_left - pad_left_decoder:-pad_right].unsqueeze(-1),
                    coarse_f[:, pad_left - pad_left_decoder + 1:1 - pad_right].unsqueeze(-1),
                ], dim=2)
                y_coarse = coarse[:, pad_left + 1:1 - pad_right]
                y_fine = fine[:, pad_left + 1:1 - pad_right]

                if self.noise_x:
                    # Randomly translate the input to the encoder to encourage
                    # translational invariance
                    total_len = coarse_f.size(1)
                    translated = []
                    for j in range(coarse_f.size(0)):
                        shift = random.randrange(256) - 128
                        translated.append(
                            noisy_f[j, pad_left - pad_left_encoder + shift:total_len - extra_pad_right + shift])
                    translated = torch.stack(translated, dim=0)
                else:
                    translated = noisy_f[:, pad_left - pad_left_encoder:]

                p_cf, vq_pen, encoder_pen, entropy, prediction = self.forward(x, translated, search_mel16,
                                                                              query_mfcc16, label)
                p_c, p_f = p_cf
                loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
                loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
                ce_loss = nn.BCELoss(nn.Sigmoid(prediction), label)

                encoder_weight = 0.01 * min(1, max(0.1, step / 1000 - 1))
                loss = loss_c + loss_f + vq_pen + encoder_weight * encoder_pen + ce_loss

                optimiser.zero_grad()
                if use_half:
                    optimiser.backward(loss)
                    if do_clip:
                        raise RuntimeError("clipping in half precision is not implemented yet")
                    else:
                        loss.backward()
                        if do_clip:
                            max_grad = 0
                            max_grad_name = ""
                            for name, param in self.named_parameters():
                                if param.grad is not None:
                                    param_max_grad = param.grad.data.abs().max()
                                    if param_max_grad > max_grad:
                                        max_grad = param_max_grad
                                        max_grad_name = name
                                    if 1000000 < param_max_grad:
                                        logger.log(f'Very large gradient at {name}: {param_max_grad}')
                            if 100 < max_grad:
                                for param in self.parameters():
                                    if param.grad is not None:
                                        if 1000000 < max_grad:
                                            param.grad.data.zero_()
                                        else:
                                            param.grad.data.mul_(100 / max_grad)
                            if running_max_grad < max_grad:
                                running_max_grad = max_grad
                                running_max_grad_name = max_grad_name

                            if 100000 < max_grad:
                                torch.save(self.state_dict(), "bad_model.pyt")
                                raise RuntimeError("Aborting due to crazy gradient (model saved to bad_model.pyt)")
                            optimiser.step()
                            running_loss_c += loss_c.item()
                            running_loss_f += loss_f.item()
                            running_loss_vq += vq_pen.item()
                            running_loss_vqc += encoder_pen.item()
                            running_entropy += entropy
                            running_loss_ce_label += ce_loss.item()
                        self.after_update()

                        speed = (i + 1) / (time.time() - start)
                        avg_loss_c = running_loss_c / (i + 1)
                        avg_loss_f = running_loss_f / (i + 1)
                        avg_loss_vq = running_loss_vq / (i + 1)
                        avg_loss_vqc = running_loss_vqc / (i + 1)
                        avg_entropy = running_entropy / (i + 1)
                        avg_loss_ce = running_loss_ce_label / (i + 1)

                        step += 1
                        k = step // 1000

                        # // track cross entropy loss as well
                        logger.status(
                            f'Epoch: {e + 1}/{epochs} -- Batch: {i + 1}/{iters} -- Loss: c={avg_loss_c:#.4} '
                            f'ce_label_loss={avg_loss_ce:#.4} f={avg_loss_f:#.4} vq={avg_loss_vq:#.4} '
                            f'vqc={avg_loss_vqc:#.4} -- Entropy: {avg_entropy:#.4} -- Grad: '
                            f'{running_max_grad:#.1} {running_max_grad_name} Speed: {speed:#.4} steps/sec -- Step: {k}k ')
                    os.makedirs(paths.checkpoint_dir, exist_ok=True)
                    torch.save(self.state_dict(), paths.model_path())
                    np.save(paths.step_path(), step)
                    logger.log_current_status()
                    logger.log(f' <saved>; w[0][0] = {self.overtone.wavernn.gru.weight_ih_l0[0][0]}')
                    if k > saved_k + 50:
                        torch.save(self.state_dict(), paths.model_hist_path(step))
                        saved_k = k
                        self.do_generate(paths, step, optimiser, dataset.path, valid_index)


    def do_generate(self, paths, step, optimizer, dataset, test_index, model_, pooling_model, deterministic=False, use_half=False,
                    verbose=False):
        k = step // 1000
        index = 1
        trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_unknownspeaker_samples(batch), batch_size=4, num_workers=4, shuffle=True, pin_memory=True)
        iters = len(trn_loader)
        loss_val = 0.
        epoch = 20
        for e in range(epoch):
            for i, (wav, query_mfcc, label) in enumerate(trn_loader):
                model_.train()
                pooling_model.train()
                wav = wav.cuda()
                query_mfcc16 = query_mfcc.cuda()
                label = label.float().cuda()

                os.makedirs(paths.gen_path(), exist_ok=True)
                #model_ = QueryEncoder()
                query_encoded = model_(query_mfcc16) #[0:1000,:]


                out = self.forward_QbE(wav,
                                       query_encoded[:, -1, :], pooling_model, verbose=verbose, use_half=use_half, index=index)

                # loss = F.binary_cross_entropy(out, label)

                loss = F.soft_margin_loss(out, label)

                loss_val += loss.item()
                if i % 20 == 1:
                    print(" VQVAE: Loss: ", loss.item(), " at iteration ", i, " labels are ", label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # batchsize
                index += 4


# # If you have wavefiles in sampling rate other than 22.5Khz, change this in utils/dsp.py
#
# # Prepare data:
# python3.6 preprocess_multispeaker.py src_location tgt_location
#
# # Train the model ( VQVAE for Spoken Term)
# python 3.6 wavernn.py

