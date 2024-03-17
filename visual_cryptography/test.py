from color_hvc import evcs_decrypt, evcs_encrypt

vc_scheme = (2, 2)
resolution = (128, 128)

cover_imgs = ["Lena.png", "Baboon.png", "Barbara.bmp", "House.bmp"]

evcs_encrypt(vc_scheme, resolution, "test_inp/Me.jpeg", "test_out", "cover_imgs", cover_imgs)

evcs_decrypt("test_out", "test_out/recover.png", vc_scheme, resolution)