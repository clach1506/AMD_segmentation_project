"""
This script is written to add optionnal postprocessing. The function png to mat transforms a folder of T
binary segmented images of size HxW into a .mat file of shape (T,H,W) under the key "MASK". 


"""
#Test on a folder of segmented images
if __name__ == "__main__":

    input_folder = "/Users/clarachoukroun/DMLA_project/test_data/BIG"
    output_matfile = "/Users/clarachoukroun/DMLA_project/test_data/BIG/COM.mat"
    #pngs_to_mat(input_folder, output_matfile)
 

"""# Exemple d'utilisation
if __name__ == "__main__":
    out = make_mask_monotone_to_mat(
        "/Users/clarachoukroun/DMLA_project/data/MATFILES-manu/020_BU_E/IR OG/COM.mat",
        mat_out=None,          # -> .../COM_corrige.mat
        var_name="MASK",
        keep_layout=True,      # conserve (H,W,T) si c'était le cas
        as_uint8=True          # écrit 0/1 en uint8 ; mettre False pour logical
    )
    print("Écrit :", out)
"""