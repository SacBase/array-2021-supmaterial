SAC2C := sac2c_d
# it is 'safe' to keep `-mt_bind` and `-cuda_arch` as these only have
# an effect when using MT or CUDA backends
SACFLAGS += -g -v1 -O3 -mt_bind simple -cuda_arch sm35
SAC2C_CALL = $(SAC2C) $(SACFLAGS)

# uppcase function
UC = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')

.PHONY: all quick cnn_mnist cnn_emnist kaggle_mnist kaggle_emnist
.SECONDARY:

all: cnn_mnist cnn_emnist kaggle_mnist kaggle_emnist
quick: cnn_mnist_seq cnn_emnist_seq kaggle_mnist_seq kaggle_emnist_seq

cnn_mnist: cnn_mnist_seq cnn_mnist_mt cnn_mnist_cuda cnn_mnist_cuda_reg
cnn_emnist: cnn_emnist_seq cnn_emnist_mt cnn_emnist_cuda cnn_emnist_cuda_reg
kaggle_mnist: kaggle_mnist_seq kaggle_mnist_mt kaggle_mnist_cuda kaggle_mnist_cuda_reg
kaggle_emnist: kaggle_emnist_seq kaggle_emnist_mt kaggle_emnist_cuda kaggle_emnist_cuda_reg

# targets to build binaries
cnn_%_seq: zhang.sac host/seq/libmnistMod.so host/seq/libcnnMod.so host/seq/libconvolutionMod.so host/seq/libactivationMod.so host/seq/libpoolingMod.so
	$(SAC2C_CALL) -t seq -noEMRCI -D$(call UC,$*) -o $@ $<

cnn_%_mt: zhang.sac host/mt-pth/libmnistMod.so host/mt-pth/libcnnMod.so host/mt-pth/libconvolutionMod.so host/mt-pth/libactivationMod.so host/mt-pth/libpoolingMod.so
	$(SAC2C_CALL) -minmtsize 50 -t mt_pth -noEMRCI -D$(call UC,$*) -o $@ $<

cnn_%_cuda: zhang.sac host/cuda/libmnistMod.so host/cuda/libcnnMod.so host/cuda/libconvolutionMod.so host/cuda/libactivationMod.so host/cuda/libpoolingMod.so
	$(SAC2C_CALL) -t cuda -noEMRCI -D$(call UC,$*) -o $@ $<

# we disable CUADE because of error in rewrite
cnn_%_cuda_reg: zhang.sac host/cuda-reg/libmnistMod.so host/cuda-reg/libcnnMod.so host/cuda-reg/libconvolutionMod.so host/cuda-reg/libactivationMod.so host/cuda-reg/libpoolingMod.so
	$(SAC2C_CALL) -t cuda_reg -doEMRCI -doCUAD -noCUADE -D$(call UC,$*) -o $@ $<

kaggle_%_seq: kaggle-1fc.sac host/seq/libmnistMod.so host/seq/libcnnMod.so host/seq/libconvolutionMod.so host/seq/libactivationMod.so host/seq/libpoolingMod.so
	$(SAC2C_CALL) -t seq -noEMRCI -D$(call UC,$*) -o $@ $<

kaggle_%_mt: kaggle-1fc.sac host/mt-pth/libmnistMod.so host/mt-pth/libcnnMod.so host/mt-pth/libconvolutionMod.so host/mt-pth/libactivationMod.so host/mt-pth/libpoolingMod.so
	$(SAC2C_CALL) -minmtsize 50 -t mt_pth -noEMRCI -D$(call UC,$*) -o $@ $<

kaggle_%_cuda: kaggle-1fc.sac host/cuda/libmnistMod.so host/cuda/libcnnMod.so host/cuda/libconvolutionMod.so host/cuda/libactivationMod.so host/cuda/libpoolingMod.so
	$(SAC2C_CALL) -t cuda -noEMRCI -D$(call UC,$*) -o $@ $<

# we disable CUADE because of error in rewrite
kaggle_%_cuda_reg: kaggle-1fc.sac host/cuda-reg/libmnistMod.so host/cuda-reg/libcnnMod.so host/cuda-reg/libconvolutionMod.so host/cuda-reg/libactivationMod.so host/cuda-reg/libpoolingMod.so
	$(SAC2C_CALL) -t cuda_reg -doEMRCI -doCUAD -noCUADE -D$(call UC,$*) -o $@ $<

# targets to build libraries
host/seq/lib%Mod.so: %.sac
	$(SAC2C_CALL) -t seq -noEMRCI $<

host/mt-pth/lib%Mod.so: %.sac
	$(SAC2C_CALL) -minmtsize 50 -noEMRCI -t mt_pth $<

host/cuda/lib%Mod.so: %.sac
	$(SAC2C_CALL) -t cuda -noEMRCI $<

host/cuda-reg/lib%Mod.so: %.sac
	$(SAC2C_CALL) -t cuda_reg -noEMRCI $<

.PHONY: clean
clean:
	$(RM) -r host tree
	$(RM) cnn_*
	$(RM) kaggle_*
