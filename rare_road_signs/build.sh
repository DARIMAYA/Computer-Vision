set -o xtrace

setup_root() {
    apt-get install -qq -y \
        ffmpeg             \
        git                \
        python3-pip        \
        python3-tk         \
        ;

    ## Unpinned
    # python3 -m pip install -qq             \
    #     albumentations                     \
    #     albumentations_experimental        \
    #     imgaug                             \
    #     kornia                             \
    #     lightning[extra]                   \
    #     matplotlib                         \
    #     moviepy                            \
    #     opencv-python-headless             \
    #     pandas                             \
    #     pytest                             \
    #     scikit-image                       \
    #     scikit-learn                       \
    #     timm                               \
    #     torch                              \
    #     torchvision                        \
    #     einops                             \
    #     ;

    ## Pinned
    python3 -m pip install -qq             \
        aiohappyeyeballs==2.6.1            \
        aiohttp==3.12.15                   \
        aiosignal==1.4.0                   \
        albucore==0.0.24                   \
        albumentations==2.0.8              \
        albumentations-experimental==0.0.1 \
        annotated-types==0.7.0             \
        antlr4-python3-runtime==4.9.3      \
        attrs==25.3.0                      \
        bitsandbytes==0.47.0               \
        certifi==2025.8.3                  \
        charset-normalizer==3.4.3          \
        contourpy==1.3.3                   \
        cycler==0.12.1                     \
        dbus-python==1.3.2                 \
        decorator==5.2.1                   \
        docstring_parser==0.17.0           \
        einops==0.8.1                      \
        filelock==3.19.1                   \
        fonttools==4.60.0                  \
        frozenlist==1.7.0                  \
        fsspec==2025.9.0                   \
        hf-xet==1.1.10                     \
        huggingface-hub==0.35.0            \
        hydra-core==1.3.2                  \
        idna==3.10                         \
        imageio==2.37.0                    \
        imageio-ffmpeg==0.6.0              \
        imgaug==0.4.0                      \
        importlib_resources==6.5.2         \
        iniconfig==2.1.0                   \
        Jinja2==3.1.6                      \
        joblib==1.5.2                      \
        jsonargparse==4.41.0               \
        jsonnet==0.21.0                    \
        kiwisolver==1.4.9                  \
        kornia==0.8.1                      \
        kornia_rs==0.1.9                   \
        lazy_loader==0.4                   \
        lightning==2.5.5                   \
        lightning-utilities==0.15.2        \
        markdown-it-py==4.0.0              \
        MarkupSafe==3.0.2                  \
        matplotlib==3.10.6                 \
        mdurl==0.1.2                       \
        moviepy==2.2.1                     \
        mpmath==1.3.0                      \
        multidict==6.6.4                   \
        networkx==3.5                      \
        numpy==2.2.6                       \
        nvidia-cublas-cu12==12.8.4.1       \
        nvidia-cuda-cupti-cu12==12.8.90    \
        nvidia-cuda-nvrtc-cu12==12.8.93    \
        nvidia-cuda-runtime-cu12==12.8.90  \
        nvidia-cudnn-cu12==9.10.2.21       \
        nvidia-cufft-cu12==11.3.3.83       \
        nvidia-cufile-cu12==1.13.1.3       \
        nvidia-curand-cu12==10.3.9.90      \
        nvidia-cusolver-cu12==11.7.3.90    \
        nvidia-cusparse-cu12==12.5.8.93    \
        nvidia-cusparselt-cu12==0.7.1      \
        nvidia-nccl-cu12==2.27.3           \
        nvidia-nvjitlink-cu12==12.8.93     \
        nvidia-nvtx-cu12==12.8.90          \
        omegaconf==2.3.0                   \
        opencv-python==4.12.0.88           \
        opencv-python-headless==4.12.0.88  \
        packaging==25.0                    \
        pandas==2.3.2                      \
        pillow==11.3.0                     \
        pluggy==1.6.0                      \
        proglog==0.1.12                    \
        propcache==0.3.2                   \
        protobuf==6.32.1                   \
        pydantic==2.11.9                   \
        pydantic_core==2.33.2              \
        Pygments==2.19.2                   \
        PyGObject==3.48.2                  \
        pyparsing==3.2.4                   \
        pytest==8.4.2                      \
        python-dateutil==2.9.0.post0       \
        python-dotenv==1.1.1               \
        pytorch-lightning==2.5.5           \
        pytz==2025.2                       \
        PyYAML==6.0.2                      \
        requests==2.32.5                   \
        rich==14.1.0                       \
        safetensors==0.6.2                 \
        scikit-image==0.25.2               \
        scikit-learn==1.7.2                \
        scipy==1.16.2                      \
        shapely==2.1.1                     \
        simsimd==6.5.3                     \
        six==1.17.0                        \
        stringzilla==4.0.13                \
        sympy==1.14.0                      \
        tensorboardX==2.6.4                \
        threadpoolctl==3.6.0               \
        tifffile==2025.9.9                 \
        timm==1.0.19                       \
        torch==2.8.0                       \
        torchmetrics==1.8.2                \
        torchvision==0.23.0                \
        tqdm==4.67.1                       \
        triton==3.4.0                      \
        typeshed_client==2.8.2             \
        typing-inspection==0.4.1           \
        typing_extensions==4.15.0          \
        tzdata==2025.2                     \
        urllib3==2.5.0                     \
        yarl==1.20.1                       \
        ;
}

setup_checker() {
    python3 --version # Python 3.12.3
    python3 -m pip freeze # see list above
    python3 -c 'import matplotlib.pyplot'
}

"$@"