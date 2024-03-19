import os
import argparse
import numpy as np
import visual_cryptography.basis_generation as basis_generation
import visual_cryptography.error_diffusion as error_diffusion
import visual_cryptography.matrices_distribution as matrices_distribution
import cv2

class ColorHVC:
    def __init__(self, vc_scheme=(2, 2), message_resolution=(128, 128)) -> None:
        if vc_scheme not in {(2, 2), (3, 4)}:
            raise ValueError(
                f"Only support (2, 2) or (3, 4) scheme, get {vc_scheme}")

        self.vc_scheme = vc_scheme
        self.num_shares = vc_scheme[1]

        self.message_resolution = message_resolution
        self.basis_matrices = basis_generation.BasisMatrices(
            vc_scheme=vc_scheme)
        self.share_resolution = self._get_share_resolution()

        self._cmy_inputs = None
        self._cmy_message = None
        self._encryption_shares = None

    def prepare(self, img_message):
        self._check_and_save_message_image(img_message)
        halftoning_message = error_diffusion.color_halftoning(
            self._cmy_message)
        self._encryption_shares = matrices_distribution.distribute_matrices(
            halftoning_message, self.basis_matrices, (*self.share_resolution, 3))

    def encrypt(self, img_inputs, img_message=None):
        if img_message is not None:
            self.prepare(img_message)
        self._check_and_save_input_image(img_inputs)

        cmy_shares = []
        for idx in range(self.num_shares):
            cmy_shares.append(error_diffusion.error_diffusion(
                self._cmy_inputs[idx], self._encryption_shares[idx],
                method='Floyd_Steinberg', threshold_modulation=True,
                scale_factor=self.basis_matrices.vip_ratio))
        rgb_shares = [conver_rgb_cmy(img) for img in cmy_shares]
        return rgb_shares

    def _check_and_save_message_image(self, img_message):
        if img_message.shape[:2] != self.message_resolution:
            img_resize = cv2.resize(
                img_message, self.message_resolution[::-1])
            self._cmy_message = conver_rgb_cmy(img_resize)
        else:
            self._cmy_message = conver_rgb_cmy(img_message)

    def _check_and_save_input_image(self, img_inputs):
        if len(img_inputs) != self.vc_scheme[1]:
            raise ValueError(
                f"number of img_inputs should be {self.vc_scheme[1]}")

        self._cmy_inputs = []
        for image in img_inputs:
            if image.shape[:2] != self.share_resolution:
                img_resize = cv2.resize(image, self.share_resolution[::-1])
                self._cmy_inputs.append(conver_rgb_cmy(img_resize))
            else:
                self._cmy_inputs.append(conver_rgb_cmy(image))

    def _get_share_resolution(self):
        expand_row = self.basis_matrices.expand_row
        expand_col = self.basis_matrices.expand_col
        share_row = self.message_resolution[0] * expand_row
        share_col = self.message_resolution[1] * expand_col
        return share_row, share_col


def conver_rgb_cmy(img_in):
    return 255 - img_in


def decrypt(img_shares):
    cmy_recover = np.zeros(img_shares[0].shape, dtype=bool)
    for image in img_shares:
        cmy_recover |= (conver_rgb_cmy(image) // 255).astype(bool)
    cmy_recover = cmy_recover.astype(np.uint8) * 255
    return conver_rgb_cmy(cmy_recover)


def evcs_encrypt(vc_scheme: tuple, resolution: tuple, input_file: str, output_file_dir: str, cover_imgs_dir: str, cover_img_names: list):
    img_inputs = [cv2.imread(os.path.join(cover_imgs_dir, ifn)) for ifn in cover_img_names]
    img_inputs = img_inputs[:vc_scheme[1]]
    img_shares = None

    if input_file:
        assert len(resolution) == 2
        assert len(vc_scheme) == 2

        num_output = vc_scheme[-1]
        output_fnames = [f"shares_{idx}.png" for idx in range(num_output)]

        img_message = cv2.imread(input_file)
        color_hvc = ColorHVC(vc_scheme=vc_scheme, message_resolution=resolution)
        img_shares = color_hvc.encrypt(img_inputs, img_message=img_message)

        for idx in range(num_output):
            out_pathname = os.path.join(output_file_dir, output_fnames[idx])
            write_status = cv2.imwrite(out_pathname, img_shares[idx])
            if not write_status:
                raise ValueError(f"cannot write to {out_pathname}")
            
def evcs_decrypt(shares_dir: str, output_file: str, vc_scheme: tuple, resolution: tuple):
    img_shares = [cv2.imread(os.path.join(shares_dir, ifn)) for ifn in os.listdir(shares_dir)]

    img_recover = decrypt(img_shares)
    cv2.imwrite(output_file, img_recover)


if __name__ == "__main__":
    vc_scheme = (2, 2)
    input_path = "src_image"
    output_path = "output_image"
    input_fnames = ["Lena.png", "Baboon.png", "Barbara.bmp", "House.bmp"]
    img_inputs = [cv2.imread(os.path.join(input_path, ifn)) for ifn in input_fnames]
    img_shares = None

    args_message = "peppers.png"
    args_resolution = (128, 128)
    args_decrypt = True
    args_output_decrypt_fname = "recover.png"

    if args_message:
        assert len(args_resolution) == 2
        assert len(vc_scheme) == 2

        num_output = vc_scheme[-1]
        output_fnames = [f"shares_{idx}.png" for idx in range(num_output)]

        img_message = cv2.imread(os.path.join(input_path, args_message))
        color_hvc = ColorHVC(vc_scheme=vc_scheme, message_resolution=args_resolution)
        img_shares = color_hvc.encrypt(img_inputs, img_message=img_message)

        for idx in range(num_output):
            out_pathname = os.path.join(output_path, output_fnames[idx])
            write_status = cv2.imwrite(out_pathname, img_shares[idx])
            if not write_status:
                raise ValueError(f"cannot write to {out_pathname}")

    if args_decrypt:
        if args_output_decrypt_fname:
            recover_fname = args_output_decrypt_fname
        else:
            recover_fname = 'message_recover.png'

        if img_shares is None:
            img_shares = img_inputs.copy()

        img_recover = decrypt(img_shares)
        cv2.imwrite(os.path.join(output_path, recover_fname), img_recover)
