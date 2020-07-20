import torch


def compute_edge_mask(image, sigma, color=True):

    if color:
        left_0 = image[:, 0, :-1, :]
        right_0 = image[:, 0, 1:, :]
        top_0 = image[:, 0, :, :-1]
        bottom_0 = image[:, 0, :, 1:]

        left_1 = image[:, 1, :-1, :]
        right_1 = image[:, 1, 1:, :]
        top_1 = image[:, 1, :, :-1]
        bottom_1 = image[:, 1, :, 1:]

        left_2 = image[:, 2, :-1, :]
        right_2 = image[:, 2, 1:, :]
        top_2 = image[:, 2, :, :-1]
        bottom_2 = image[:, 2, :, 1:]

        mask_h = torch.exp(-1 * ( (left_0 - right_0) ** 2 +(left_1 - right_1) ** 2+(left_2 - right_2) ** 2 )/ (2 * sigma ** 2))
        mask_v = torch.exp(-1 * ( (top_0 - bottom_0) ** 2 +  (top_1 - bottom_1) ** 2 + (top_2 - bottom_2) ** 2) / (2 * sigma ** 2))

    else:
        image_dims = list(image.size())

        if image_dims[1] > 1:
            image = torch.sum(image,dim=1)

        left_ = image[:, :-1, :]
        right_ = image[:, 1:, :]
        top_ = image[:, :, :-1]
        bottom_ = image[:, :, 1:]

        mask_h = torch.exp(-1 * (left_ - right_) ** 2 / (2 * sigma ** 2))
        mask_v = torch.exp(-1 * (top_ - bottom_) ** 2 / (2 * sigma ** 2))

    return mask_h, mask_v


def compute_edge_mask_diag(image, sigma):

    image_dims = list(image.size())

    if image_dims[1] > 1:
        image = torch.sum(image,dim=1)

    left_ = image[:, :-1, :]
    left_ = left_[:,:,1:]
    diag1 = image[:, 1:, :]
    diag1 = diag1[:,:, :-1]

    top_ = image[:, :, :-1]
    top_ = top_[:,:-1,:]
    diag2 = image[:, :, 1:]
    diag2 = diag2[:,1:,:]

    mask_d1 = torch.exp(-1 * (left_ - diag1) ** 2 / (2 * sigma ** 2))*0.707
    mask_d2 = torch.exp(-1 * (top_ - diag2) ** 2 / (2 * sigma ** 2))*0.707

    return mask_d1, mask_d2


def regularized_loss_per_channel_diag(mask_d1, mask_d2, cl, prediction, true_class, negative_class=False):

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

        if prediction is None:
            return 0.
        else:
            mask_d1 = extract_needed_mask(mask_d1, true_class, cl, extract_condition_equal_fn)
            mask_d2 = extract_needed_mask(mask_d2, true_class, cl, extract_condition_equal_fn)

    left = prediction[:,cl ,:-1,:]
    left = left[:,:,1:]
    diag1 = prediction[:, cl , 1:, :]
    diag1 = diag1[:, :, :-1]

    top = prediction[:,cl ,:,:-1]
    top = top[:, :-1,:]
    diag2 = prediction[:,cl ,:,1:]
    diag2 = diag2[:,1:,:]


    h = torch.mean(abs(left - diag1) * mask_d1)
    v = torch.mean(abs(top - diag2) * mask_d2)

    return((h + v)/2.0)


def  regularized_loss_per_channel(mask_h, mask_v, cl, prediction, true_class, negative_class=False):

    if negative_class:
        prediction = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

        if prediction is None:
            return 0.
        else:
            mask_h = extract_needed_mask(mask_h, true_class, cl)
            mask_v = extract_needed_mask(mask_v, true_class, cl)

    left = prediction[:, cl ,:-1, :]
    right = prediction[:, cl , 1:, :]
    top = prediction[:,cl ,:,:-1]
    bottom = prediction[:,cl ,:,1:]

    h = torch.mean(abs(left - right) * mask_h)
    v = torch.mean(abs(top - bottom) * mask_v)

    return((h + v)/2.0)



def middle_sq_loss_per_channel(cl, prediction, true_class, square_w):

    prediction_pos = extract_needed_predictions(true_class, prediction, cl, extract_condition_equal_fn)

    if prediction_pos is not None:
        height = prediction_pos.size()[2]
        width = prediction_pos.size()[3]

        middleH = height // 2
        middleW = width // 2

        middle_sq  = torch.mean(prediction_pos[:, cl, middleH-square_w:middleH+square_w,middleW-square_w:middleW+square_w])

        loss = (middle_sq - 1.0)  ** 2
    else:
        loss = 0.

    prediction_neg = extract_needed_predictions(true_class, prediction, cl, extract_condition_not_equal_fn)
    if prediction_neg is not None:
        loss = loss + (torch.mean(prediction_neg[:, cl, :, :])) ** 2

    return loss


def extract_needed_predictions(true_class, y_pr, cl, cond_fn):

    which_class = cond_fn(true_class, cl)

    needed_prediction = y_pr[which_class, :, :, :]
    if needed_prediction.size()[0] == 0:
        needed_prediction = None

    return needed_prediction


def extract_condition_equal_fn(true_class, cl):
    return true_class == cl


def extract_condition_not_equal_fn(true_class, cl):
    return true_class != cl



def extract_needed_mask(mask, true_class,  cl):

    which_class = (true_class == cl)

    needed_mask = mask[which_class, :, :]
    if needed_mask.size()[0] == 0:
        needed_mask = None

    return needed_mask


