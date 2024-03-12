from physion.dataviz.camera import *

from physion.pupil import roi, process


def imgPupil_process(img, args, exp=0.5,
                    bounds=[0.05, 0.75]):
    Img = (img-np.min(img))/(np.max(img)-np.min(img))
    Img = np.power(Img, exp) 
    return Img[args['PupilZoom'][0]:args['PupilZoom'][2],\
               args['PupilZoom'][1]:args['PupilZoom'][3]] 


def add_pupil_fit(metadata, ax, index):

    cx, sx = metadata['pupil_cx'][index], metadata['pupil_sx'][index]
    cy, sy = metadata['pupil_cy'][index], metadata['pupil_sy'][index]

    cy -= (metadata['PupilZoom'][0]-metadata['pupil_ymin'])
    cx -= (metadata['PupilZoom'][1]-metadata['pupil_xmin'])

    # /!\ weird why this works...
    coords = [cy, cx, sx, sy]
    pupil_fit = process.ellipse_coords(*coords, transpose=True)
    return ax.plot(pupil_fit[0], pupil_fit[1], '.', 
                       markersize=1, color='red')

def show_pupil(metadata, 
               t=0, 
               ax=None,
               zoom=0.):

    t_index = np.argmin((metadata['raw_Face_times']-t)**2)

    if ax is None:
        _, ax = plt.subplots(1)
        ax.axis('equal')
        ax.axis('off')

    if 'PupilZoom' not in metadata:
        xmin, xmax = metadata['pupil_xmin'], metadata['pupil_xmax']
        ymin, ymax = metadata['pupil_ymin'], metadata['pupil_ymax']
        dx, dy = xmax-xmin, ymax-ymin
        metadata['PupilZoom'] = [int(ymin-zoom/2.*dy),
                                 int(xmin-zoom/2.*dx),
                                 int(ymax+zoom/2.*dy),
                                 int(xmax+zoom/2.*dx)]

    # find boundaries
    img = np.load(metadata['raw_Face_FILES'][t_index])

    Img = imgPupil_process(img, metadata)

    imshow = ax.imshow(Img, cmap='gray')

    add_pupil_fit(metadata, ax, index=t_index)

    return imshow

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("raw_data_folder", type=str)

    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity", 
                        action="store_true")

    args = vars(parser.parse_args())

    metadata = np.load(os.path.join(args['raw_data_folder'],
                                    'metadata.npy'),
                       allow_pickle=True).item()

    loadCameraData(metadata, args['raw_data_folder'])

    show_pupil(metadata, t=120)
    show_pupil(metadata, t=150)
    plt.show()

 


