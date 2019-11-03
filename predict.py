from tqdm import tqdm

from unet.unet_model import *
from utils import *
from utils.data_vis import plot_img_and_mask


def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_gpu=True):
    net.eval()

    img = resize_and_crop(full_img, scale=scale_factor).astype(np.float32)
    img = np.transpose(normalize(img), (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(dim=0)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        mask = net(img)
        mask = torch.sigmoid(mask).squeeze().cpu().numpy()

    return mask > out_threshold


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    scale, mask_threshold, cpu,  viz, no_save = 1, 0.5, False, False, False
    # model: 'Unet', 'Res_Unet', 'Ringed_Res_Unet'
    network = 'Ringed_Res_Unet'

    img = Image.open('your_test_img.png')
    model = 'result/logs/test.pkl'

    if network == 'Unet':
        net = Unet(n_channels=3, n_classes=1)
    elif network == 'Res_Unet':
        net = Res_Unet(n_channels=3, n_classes=1)
    else:
        net = Ringed_Res_Unet(n_channels=3, n_classes=1)

    if not cpu:
        net.cuda()
        net.load_state_dict(torch.load(model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=scale,
                       out_threshold=mask_threshold,
                       use_gpu=not cpu)

    if viz:
        print("Visualizing results for image {}, close to continue ...".format(j))
        plot_img_and_mask(img, mask)

    if not no_save:
        result = mask_to_image(mask)

        if network == 'Unet':
            result.save('predict_u.png')
        elif network == 'Res_Unet':
            result.save('predict_ru.png')
        else:
            result.save('predict_rru.png')