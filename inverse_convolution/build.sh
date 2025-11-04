set -o xtrace

setup_root() {
    apt-get install -qq -y \
        python3-pip        \
        python3-tk         \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     matplotlib                         \
    #     numpy                              \
    #     pillow                             \
    #     pytest                             \
    #     scikit-image                       \
    #     scikit-learn                       \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        contourpy==1.3.3                   \
        cycler==0.12.1                     \
        fonttools==4.60.0                  \
        imageio==2.37.0                    \
        iniconfig==2.1.0                   \
        joblib==1.5.2                      \
        kiwisolver==1.4.9                  \
        lazy_loader==0.4                   \
        matplotlib==3.10.6                 \
        networkx==3.5                      \
        numpy==2.2.6                       \
        packaging==25.0                    \
        pillow==11.3.0                     \
        pluggy==1.6.0                      \
        Pygments==2.19.2                   \
        pyparsing==3.2.4                   \
        pytest==8.4.2                      \
        python-dateutil==2.9.0.post0       \
        scikit-image==0.25.2               \
        scikit-learn==1.7.2                \
        scipy==1.16.2                      \
        six==1.17.0                        \
        threadpoolctl==3.6.0               \
        tifffile==2025.9.9                 \
        ;
}

setup_checker() {
    python3 --version # Python 3.12.3
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"