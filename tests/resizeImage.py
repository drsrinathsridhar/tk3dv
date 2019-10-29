import cv2, math, sys, os

if __name__ == '__main__':
    ImFile = sys.argv[1]
    Image = cv2.imread(ImFile, -1)
    OutIm = Image
    Size = (224, 224) # W, H
    # Size = (1920, 1080) # W, H
    # Size = (320, 240) # W, H
    # Size = (1100, 500) # W, H
    # Size = (600, 100) # W, H

    if Size is not None:
        # Check if the aspect ratios are the same
        OrigSize = Image.shape[:-1]
        OrigAspectRatio = OrigSize[1] / OrigSize[0] # W / H
        ReqAspectRatio = Size[0] / Size[1] # W / H # CAUTION: Be aware of flipped indices
        print(OrigAspectRatio)
        print(ReqAspectRatio)
        if math.fabs(OrigAspectRatio-ReqAspectRatio) > 0.01:
            # Different aspect ratio detected. So we will be fitting the smallest of the two images into the larger one while centering it
            # After centering, we will crop and finally resize it to the request dimensions
            if ReqAspectRatio < OrigAspectRatio:
                NewSize = [OrigSize[0], int(OrigSize[0] * ReqAspectRatio)] # Keep height
                Center = int(OrigSize[1] / 2) - 1
                HalfSize = int(NewSize[1] / 2)
                Image = Image[:, Center-HalfSize:Center+HalfSize, :]
            else:
                NewSize = [int(OrigSize[1] / ReqAspectRatio), OrigSize[1]] # Keep width
                Center = int(OrigSize[0] / 2) - 1
                HalfSize = int(NewSize[0] / 2)
                Image = Image[Center-HalfSize:Center+HalfSize, :, :]
        OutIm = cv2.resize(Image, dsize=Size, interpolation=cv2.INTER_CUBIC)

    print('Output image size:', OutIm.shape)
    cv2.imwrite('out.png', OutIm)

