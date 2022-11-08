import json, os

def set_filename_and_folder(self):
    self.filename = generate_filename_path(self.root_datafolder,
                                filename='metadata',
                        extension='.npy',
                with_FaceCamera_frames_folder=self.metadata['FaceCamera'])
    self.datafolder.set(os.path.dirname(self.filename))


def save_experiment(self, metadata):
    # SAVING THE METADATA FILES
    metadata['filename'] = str(self.datafolder.get())
    for key in self.protocol:
        metadata[key] = self.protocol[key]
    np.save(os.path.join(str(self.datafolder.get()), 'metadata.npy'), metadata)
    print('[ok] Metadata data saved as: %s ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))
    self.statusBar.showMessage('Metadata saved as: "%s" ' % os.path.join(str(self.datafolder.get()), 'metadata.npy'))

