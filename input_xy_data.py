import numpy as np

class xy_data(object):
    def read_in(self,seed):
        self._index=0
        self._seed=seed
        print("seed= " + str(self._seed))

        self._train_good_orig = np.loadtxt("train_good.txt",skiprows=0)
        self._Y_train_good = np.array([np.ones([self._train_good_orig.shape[0]]) , np.zeros([self._train_good_orig.shape[0]])]).T
        print("Train good data shape: " + str(self._train_good_orig.shape) + " , Y shape: " + str(self._Y_train_good.shape))

        self._test_good_orig = np.loadtxt("test_good.txt",skiprows=0)
        self._Y_test_good = np.array([np.ones([self._test_good_orig.shape[0]]) , np.zeros([self._test_good_orig.shape[0]])]).T
        print("Test good data shape: " + str(self._test_good_orig.shape) + " , Y shape: " + str(self._Y_test_good.shape))

        self._train_bad_orig = np.loadtxt("train_bad.txt",skiprows=0)
        self._Y_train_bad = np.array([np.zeros([self._train_bad_orig.shape[0]]) , np.ones([self._train_bad_orig.shape[0]])]).T
        print("Train bad data shape: " + str(self._train_bad_orig.shape) + " , Y shape: " + str(self._Y_train_bad.shape))

        self._test_bad_orig = np.loadtxt("test_bad.txt",skiprows=0)
        self._Y_test_bad = np.array([np.zeros([self._test_bad_orig.shape[0]]) , np.ones([self._test_bad_orig.shape[0]])]).T
        print("Test bad data shape: " + str(self._test_bad_orig.shape) + " , Y shape: " + str(self._Y_test_bad.shape))


    def shape(self,ndim):
        ndim=np.maximum(ndim, 34)
        assert((ndim-30)%2==0)
        print("Shape into " + str(ndim) + " x " + str(ndim))

        ### Train good ###
        m=self._train_good_orig.shape[0]
        D_pbg=self._train_good_orig[:,1:901]
        D_cry=self._train_good_orig[:,1001:2157]

        D_pbg=D_pbg.reshape((m,30,30))
        D_cry=D_cry.reshape((m,34,34))

        ddim=(int)((ndim-30)/2)
        D_pbg=np.lib.pad(D_pbg,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        ddim=(int)((ndim-34)/2)
        D_cry=np.lib.pad(D_cry,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        print("D_pbg shape= " + str(D_pbg.shape))
        print("D_cry shape= " + str(D_cry.shape))

        self._train_good=np.concatenate((D_cry[...,np.newaxis], D_pbg[...,np.newaxis]),axis=3)
        print("Train_good shape= " + str(self._train_good.shape))

        ### Test good ###
        m=self._test_good_orig.shape[0]
        D_pbg=self._test_good_orig[:,1:901]
        D_cry=self._test_good_orig[:,1001:2157]

        D_pbg=D_pbg.reshape((m,30,30))
        D_cry=D_cry.reshape((m,34,34))

        ddim=(int)((ndim-30)/2)
        D_pbg=np.lib.pad(D_pbg,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        ddim=(int)((ndim-34)/2)
        D_cry=np.lib.pad(D_cry,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        print("D_pbg shape= " + str(D_pbg.shape))
        print("D_cry shape= " + str(D_cry.shape))

        self._test_good=np.concatenate((D_cry[...,np.newaxis], D_pbg[...,np.newaxis]),axis=3)
        print("Test_good shape= " + str(self._test_good.shape))

        ### Train bad ###
        m=self._train_bad_orig.shape[0]
        D_pbg=self._train_bad_orig[:,1:901]
        D_cry=self._train_bad_orig[:,1001:2157]

        D_pbg=D_pbg.reshape((m,30,30))
        D_cry=D_cry.reshape((m,34,34))

        ddim=(int)((ndim-30)/2)
        D_pbg=np.lib.pad(D_pbg,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        ddim=(int)((ndim-34)/2)
        D_cry=np.lib.pad(D_cry,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        print("D_pbg shape= " + str(D_pbg.shape))
        print("D_cry shape= " + str(D_cry.shape))

        self._train_bad=np.concatenate((D_cry[...,np.newaxis], D_pbg[...,np.newaxis]),axis=3)
        print("Train_bad shape= " + str(self._train_bad.shape))

        ### Test bad ###
        m=self._test_bad_orig.shape[0]
        D_pbg=self._test_bad_orig[:,1:901]
        D_cry=self._test_bad_orig[:,1001:2157]

        D_pbg=D_pbg.reshape((m,30,30))
        D_cry=D_cry.reshape((m,34,34))

        ddim=(int)((ndim-30)/2)
        D_pbg=np.lib.pad(D_pbg,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        ddim=(int)((ndim-34)/2)
        D_cry=np.lib.pad(D_cry,((0,0),(ddim,ddim),(ddim,ddim)),'constant',constant_values=(0))
        print("D_pbg shape= " + str(D_pbg.shape))
        print("D_cry shape= " + str(D_cry.shape))

        self._test_bad=np.concatenate((D_cry[...,np.newaxis], D_pbg[...,np.newaxis]),axis=3)
        print("Test_bad shape= " + str(self._test_bad.shape))



    def out_train_good(self, ns, ne):
        return self._train_good[ns:ne], self._Y_train_good[ns:ne]

    def out_test_good(self, ns, ne):
        return self._test_good[ns:ne], self._Y_test_good[ns:ne]

    def out_train_bad(self, ns, ne):
        return self._train_bad[ns:ne], self._Y_train_bad[ns:ne]

    def out_test_bad(self, ns, ne):
        return self._test_bad[ns:ne], self._Y_test_bad[ns:ne]

    def out_test(self):
        return np.concatenate((self._test_good,self._test_bad),axis=0), np.concatenate((self._Y_test_good,self._Y_test_bad),axis=0)

    def make_batch(self,mini_batch_size=128):
        if self._seed > 0: np.random.seed(self._seed)
        self._train_X=np.concatenate((self._train_good,self._train_bad),axis=0)
        self._train_Y=np.concatenate((self._Y_train_good,self._Y_train_bad),axis=0)
        print("train_X shape: " + str(self._train_X.shape))
        print("train_Y shape: " + str(self._train_Y.shape))

        m=self._train_X.shape[0]
        assert(m==self._train_Y.shape[0])
        print("Total num of examples= " + str(m))

        self._mini_batch_size=mini_batch_size
        self._nbatch=(int)(m/mini_batch_size +1)
        self._is_complete_div=(m%mini_batch_size==0)
        print("Mini batch size= " + str(self._mini_batch_size) +  " , Num of mini batches= " + str(self._nbatch) + " , is complete= " + str(self._is_complete_div))

        perm=np.arange(m)
        np.random.shuffle(perm)
        self._perm=perm
        self._train_X=self._train_X[perm]
        self._train_Y=self._train_Y[perm]


    def out_perm(self):
        return self._perm, self._train_X, self._train_Y


    def xy_next_batch(self):
        n=(int)(self._index % self._nbatch)
        ns=(int)(n * self._mini_batch_size)

        if self._is_complete_div or n != (self._nbatch-1):
            ne=(int)((n+1) * self._mini_batch_size)
        else:
            ne=self._train_Y.shape[0]

        #print("index= " + str(self._index) + " , n= " + str(n) + " , ns= " + str(ns) + " , ne= " + str(ne))

        self._index += 1
        return self._train_X[ns:ne], self._train_Y[ns:ne]








