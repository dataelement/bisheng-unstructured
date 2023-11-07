#!/bin/bash


function install_texlive() {
    # ISO: https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/
    # ISO_FILE="/home/hanfeng/tars/texlive.iso"
    # mount -o loop $ISO_FILE /mnt
    MOUNT_PATH="/home/hanfeng/tars/texlive_iso_mirror"
    pushd ${MOUNT_PATH}
    ./install-tl -no-gui -texdir /opt/texlive/2023
    popd
    unmount /mnt
}


function install_deps() {
    apt-get install libfontconfig1
}


function update_path() {
    echo "export PATH=/opt/texlive/2023/bin/x86_64-linux:\$PATH" >> /root/.bashrc
    echo "export MANPATH=/opt/texlive/2023/texmf-dist/doc/man:\$MANPATH" >> /root/.bashrc
    echo "export INFOPATH=/opt/texlive/2023/texmf-dist/doc/info:\$INFOPATH" >> /root/.bashrc
}


function update_fonts() {
    # add tex pkg
    # mktexlsr

    EXTR_FONT_DIR="/home/hanfeng/tars/texlive_fonts"
    cp -fr ${EXTR_FONT_DIR} /usr/share/fonts/
    mkfontscale  
    mkfontdir  
    fc-cache -fsv
    # fc-cache -fv
    fc-list :lang=zh-cn
}


function install_pandoc() {
    PANDOC_TAR_FILE="/home/hanfeng/tars/pandoc-3.1.9-linux-amd64.tar.gz"
    tar zxf ${PANDOC_TAR_FILE} -C /opt/pandoc
    echo "export PATH=/opt/pandoc/bin/:\$PATH" >> /root/.bashrc
}


install_texlive
