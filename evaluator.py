import torch

class Evaluator(object):
    """
    Standard evaluation metrics
    """

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.confusion_matrix = torch.zeros((self.num_classes,)*2).to(self.device)

    def Pixel_Accuracy(self):
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc.cpu().numpy()

    def Pixel_Accuracy_Class(self):
        Acc = torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = Acc[~torch.isnan(Acc)].mean()
        return Acc.cpu().numpy()

    def Mean_Intersection_over_Union(self):
        MIoU = torch.diag(self.confusion_matrix) / (
                    torch.sum(self.confusion_matrix, axis=1) + torch.sum(self.confusion_matrix, axis=0) -
                    torch.diag(self.confusion_matrix))

        MIoU = MIoU[~torch.isnan(MIoU)].mean()
        return MIoU.cpu().numpy()

    def Jaccard(self):
        # for binary case, computes Jaccard index for object class (class 1)
        intersection = torch.diag(self.confusion_matrix)
        ground_truth_set = torch.sum(self.confusion_matrix, axis=1)
        predicted_set = torch.sum(self.confusion_matrix, axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection = intersection[1]
        union = union[1]

        jaccard = intersection/union

        return jaccard.cpu().numpy()

    def f_score(self, betaSq=0.3):

        epsilon = 1e-12

        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        fn = self.confusion_matrix[1, 0]

        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)

        f_score = (1+betaSq)*precision*recall/(betaSq*precision+recall)

        return f_score.cpu().numpy()

    def _generate_matrix(self, gt, pr):
        target_mask = (gt >= 0) & (gt < self.num_classes)
        gt = gt[target_mask]
        pr = pr[target_mask]
        gt = gt.long()

        indices = self.num_classes * gt + pr
        conf = torch.bincount(indices, minlength=self.num_classes ** 2)
        conf = conf.reshape(self.num_classes, self.num_classes)

        return conf.float()

    def add_batch(self, gt, pr):
        if len(list(pr.size())) == 4:  # convert from prob to labels
            _, pr = torch.max(pr, dim=1)

        assert gt.shape == pr.shape
        self.confusion_matrix += self._generate_matrix(gt, pr)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes,)*2).to(self.device)

