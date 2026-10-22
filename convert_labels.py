import os
import nibabel as nib
import numpy as np

label_dir = os.path.join(
    os.environ["nnUNet_raw"],
    "Dataset123_Brain",
    "labelsTr"
)

filenames = sorted(os.listdir(label_dir))

count = 0

for filename in filenames:
    if not filename.endswith(".nii.gz"):
        continue

    path = os.path.join(label_dir, filename)

    img = nib.load(path)
    data = np.asanyarray(img.dataobj).copy()

    labels_before = np.unique(data)

    data[data == 4] = 3

    labels_after = np.unique(data)

    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, path)

    count += 1

    print(
        count,
        filename,
        labels_before,
        "->",
        labels_after
    )

print("Finished.")
print("Files processed:", count)
