# Installation Troubleshooting / Common Issues

- the `PyQt` package can be broken after those steps, re-start from a fresh install with `pip uninstall PyQt5` and `pip install PyQt5`. 
- In linux, the `libqxcb.so` binding is making problems, this can be solved by deleting the following file: `rm ~/miniconda3/lib/python3.11/site-packages/cv2/qt/plugins/platforms/libqxcb.so`.
- In linux, there can be a `krb5` version mismatch between Qt and Ubuntu packages. Download the latest on [the kerboeros website](https://web.mit.edu/kerberos/) and install it from source with: `tar xf krb5-1.18.2.tar.gz; cd krb5-1.18.2/src; ./configure --prefix=/opt/krb5/ ; make && sudo make install`. Then do the binding with: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/krb5/lib` (you can put this in your `~/.bashrc`).
