__author__ = 'yihanjiang'

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets, weight_mask = None):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', weight=weight_mask)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none', weight=weight_mask)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss


        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def customized_loss(output, X_train, args, size_average = True, noise = None, code = None):

    if args.loss == 'bce':
        output = torch.clamp(output, 0.0, 1.0)
        if size_average == True:
            loss = F.binary_cross_entropy(output, X_train)
        else:
            return [F.binary_cross_entropy(item1, item2) for item1, item2 in zip(output, X_train)]

    ##########################################################################################
    # The result are all experimental, nothing works..... BCE works well.
    ##########################################################################################
    elif args.loss == 'soft_ber':
        output = torch.clamp(output, 0.0, 1.0)
        loss = torch.mean(((1.0 - output)**X_train )* ((output) ** (1.0-X_train)))
        #print(loss)

    elif args.loss == 'bce_rl':
        output = torch.clamp(output, 0.0, 1.0)

        bce_loss = F.binary_cross_entropy(output, X_train, reduction='none')

        # support different sparcoty of feedback noise.... for future....
        ber_loss      = torch.ne(torch.round(output), torch.round(X_train)).float()
        baseline      = torch.mean(ber_loss)
        ber_loss      = ber_loss - baseline

        loss = args.ber_lambda * torch.mean(ber_loss*bce_loss) + args.bce_lambda * bce_loss.mean()

    elif args.loss == 'enc_rl':  # binary info about if decoding is wrong or not.
        output = torch.clamp(output, 0.0, 1.0).detach()
        ber_loss      = torch.ne(torch.round(output), torch.round(X_train)).float().detach()

        # baseline      = torch.mean(ber_loss)
        # ber_loss      = ber_loss - baseline
        item = ber_loss*torch.abs(code)
        loss          = torch.mean(ber_loss*torch.abs(code))

    elif args.loss == 'bce_block':
        output = torch.clamp(output, 0.0, 1.0)
        BCE_loss_tmp = F.binary_cross_entropy(output, X_train, reduction='none')
        tmp, _ = torch.max(BCE_loss_tmp, dim=1, keepdim=False)
        max_loss     = torch.mean(tmp)
        loss = max_loss

    elif args.loss == 'focal':
        output = torch.clamp(output, 0.0, 1.0)
        loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)(output, X_train)

    elif args.loss == 'mse':
        output = torch.clamp(output, 0.0, 1.0)
        output_logit = torch.log(output/(1.0 - output+1e-10))
        loss = F.mse_loss(output_logit, X_train)

    elif args.loss == 'maxBCE':
        output = torch.clamp(output, 0.0, 1.0)
        BCE_loss_tmp = F.binary_cross_entropy(output, X_train, reduce=False)

        bce_loss = torch.mean(BCE_loss_tmp)
        pos_loss = torch.mean(BCE_loss_tmp, dim=0)

        tmp, _ = torch.max(pos_loss, dim=0)
        max_loss     = torch.mean(tmp)

        loss = bce_loss + args.lambda_maxBCE * max_loss

    elif args.loss == 'sortBCE':
        output = torch.clamp(output, 0.0, 1.0)
        BCE_loss_tmp = F.binary_cross_entropy(output, X_train, reduce=False)

        bce_loss = torch.mean(BCE_loss_tmp)
        pos_loss = torch.mean(BCE_loss_tmp, dim=0)

        tmp, _ = torch.sort(pos_loss, dim=-1, descending=True, out=None)

        sort_loss     = torch.sum(tmp[:5, :])

        loss = bce_loss + args.lambda_maxBCE * sort_loss

    return loss

