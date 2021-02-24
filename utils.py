
# #########################################################################
# This file includes code that is part of mesh manipulation,
# data loading and visualization functionality for fatigue computations
# using the BECAS cross-section analysis matlab software.
# #########################################################################
# import wm_wmblade as wm
import numpy as np
import matplotlib.pyplot as pplot


## Functions for plotting fatigue values on a cross-section of the blade:
default_mesh_file = 'data/wmblade4_mat/cross_section0.npz'

class BladeCSMeshPlotter(object):
    def __init__(self,mesh_file, mesh_finite_mask = None):
        self.mesh_finite_mask = mesh_finite_mask
        self.mesh_file = mesh_file
        self.mesh = BECASMesh.make_BECAS_mesh_from_file(mesh_file)

    def plot_value(self,value):
        """
        plots a value on the mesh
        """
        #code.interact(local = dict(locals(), globals()))
        val = np.zeros(self.mesh_finite_mask.shape); 
        val[self.mesh_finite_mask] = value;
        self.mesh.plot_mesh_value(val = val)

class PlotFatvals:
    def __init__(self, theta_el, r0):
        """
        Plotting functionality for cross-section fatigue values.
        """

        self.x0 = r0 * np.sin(theta_el)
        self.y0 = r0 * np.cos(theta_el)
        self.r0 = r0;
        self.theta_el = theta_el
        self.xx = []
        self.yy = []
        self.color = []

    @staticmethod
    def scaling_function( val):
        return val

    @staticmethod
    def color_scaling_function(val):
        return np.abs(val)**0.3 * np.sign(val);

    def scatter_vals(self, data, indices):
        """
        collect the values for plotting the scatterplots (fatigue data)
        """
        self.xx =  (self.r0 + self.scaling_function(data[indices,:])) * np.sin(self.theta_el)
        self.yy = (self.r0 + self.scaling_function(data[indices,:]) ) * np.cos(self.theta_el)
        self.color=  self.color_scaling_function( self.scaling_function(data[indices,:]))

    def make_plot(self,data, inds, opacity = 0.05, tight = True, color = None , marker_config = {'size':None,'shape':None,'cs_color' : 'black'}, ax = None, minmax_col = None):
        """
        plots fatigue values on the cross-section.

        parameters:
          data: the actual fatigue data to be plotted
          inds: the non-zero indices corresponding to the cross-section value. This mask is taken from the generative model (or from BladeCSMeshPlotter() object, since it is needed there for pre-processing the dataset.
          opacity (optional): opacity of the plotted values.
          tight (optional): keep aspect ratio
          color(optional): if "None" then a default color-scaling function is used.
        """
        self.scatter_vals(data,inds)
        plt_artist = pplot if ax is None else ax

        plt_artist.scatter(self.x0,self.y0, c = 'black')
        if color is not None:
            color_ = color
        else:
            color_ = self.color

        
        plt_artist.scatter(self.xx, self.yy , c = color_, alpha = opacity,marker = marker_config['shape'], s=marker_config['size'])
            

        if tight:
            plt_artist.gca().set_xlim([-2.,2.])
            plt_artist.gca().set_ylim([-2.,2.])


## Functions for pre/processing the blade fatigue dataset:

normalization_strategy = {
        'power0.1' : {'forward' : lambda x : np.power(x,0.1) , 'inverse' : lambda x : np.power(x,10)},
        'log' : {'forward' : lambda x : np.log(x) ,'inverse' : lambda x : np.exp(x)}
        }

class DELDatasetPreProc(object):
    def __init__(self, normalization_strategy_string = 'power0.1', data_folder = 'data/11apr19', fname_inputs = 'inputs_runJan19.npz'):
        """
        loads the blade cross-section fatigue dataset
        """

        # The following dataset has the blade root fatigue computations
        # for a 1.5MW blade.
        self.raw_data = np.load('%s/DEL_results_python.npy'%data_folder);
        self.case_numbers_run = np.load('%s/case_numbers_run.npy'%data_folder);
        # load the inputs for the case numbers available:
        data_input = np.load('%s/%s'%(data_folder, fname_inputs))
        self.data_input_header = data_input['header']
        self.data_input = data_input['data']
        cases = self.case_numbers_run.astype(np.int32)

        # may lead to bug - check for consistency if something down the line seems out of place!
        self.data_input  = self.data_input[cases,1:].squeeze();

        self.ncases = len(self.raw_data)
        self.vector_size = self.raw_data[0].shape
        self.normalization_string = normalization_strategy_string
        self.NORMALIZATION_APPLIED_DEL = False
        self.NORMALIZATION_APPLIED_X = False

    def scaling_forward(self, data):
        """
        the data are badly scaled. Therefore they are transformed with a simple invertible function
        in order to make it easy to compute with them.
        """
        return normalization_strategy[self.normalization_string]['forward'](data)

    def scaling_inverse(self, data):
        """
        for getting back to the original data from the transformed.
        """
        return normalization_strategy[self.normalization_string]['inverse'](data)

    def apply_normalization_X(self):
        if not self.NORMALIZATION_APPLIED_X:
            self.data_input_mean = np.mean(self.data_input,0)
            self.data_input_std = np.std(self.data_input,0)
            self.data_input_normalized = (self.data_input-self.data_input_mean) / self.data_input_std
            self.NORMALIZATION_APPLIED_X = True
        
    def apply_normalization_DEL(self):
        """
        as usual, transform the data so that they are not badly scaled.
        
        The NaNs and Infs are removed and their indices are kept.
        The data is raised to a power smaller than 1 (for example 0.1) to squash the large values.
        The goal is to make the data have std 1 without becoming negative.
        """
        if not self.NORMALIZATION_APPLIED_DEL:
            self.hasval_inds = np.isfinite(np.sum(self.raw_data[0],1)) * (np.sum(self.raw_data[0],1) != 0);
            
            data = [np.sum(k,1) for k in self.raw_data]
            data = np.vstack(data);
            
            data = self.scaling_forward(data[:,self.hasval_inds])
            self.scaled_data_mean = np.mean(data,0)
            self.scaled_data_std  =np.std(data,0)
            self.normalized_data = (data - self.scaled_data_mean) / self.scaled_data_std
            self.NORMALIZATION_APPLIED_DEL = True

    def apply_normalization(self):
        self.apply_normalization_DEL()
        self.apply_normalization_X()
        
    def get_normalized_data_DEL(self):
        """
        After the normalization has been applied, use this to return the 
        normalized data.
        """
        self.apply_normalization_DEL() # does nothing if data is already normalized
        return self.normalized_data

    def get_normalized_data_X(self):
        self.apply_normalization_X()
        return self.data_input_normalized

    def unnormalize_DEL(self,in_data):
        """
        returns an un-normalized version of the dataset.
        """
        return self.scaling_inverse((in_data + self.scaled_data_mean) * self.scaled_data_std)

    def unnormalize_X(self,in_data_X):
        return (in_data_X * self.data_input_std) + self.data_input_mean
        

    def train_test_split(self,pct_train = 0.8, seed = 100, permute = False):
        """
        splits the dataset to training and testing set.
        """
        if not self.NORMALIZATION_APPLIED_X or not self.NORMALIZATION_APPLIED_DEL:
            raise AssertionError('You must first normalize your dataset.')

        self.perminds = None
        if permute:
            perminds = np.random.permutation(self.raw_data.shape[0])
            self.perminds = perminds
            data = self.normalized_data[perminds,:]
            input_data = self.data_input_normalized[perminds,:]
        else:
            data = self.normalized_data
            input_data = self.data_input_normalized

        split_train = int(self.normalized_data.shape[0] * pct_train)
        train_DEL = data[:split_train,:]
        test_DEL = data[split_train:,:]
        train_X = input_data[:split_train,:]
        test_X = input_data[split_train:,:]
        dataset = {'train' : {'DEL':train_DEL,'X' : train_X}, 'test' : {'DEL' : test_DEL, 'X': test_X},'seed':seed}
        return dataset


    def _reshape_in_minibatches(self,Xdata,Ydata, batch_size):
        """
        reshapes in minibatches.
        """
        def get_minibatch_list(data):
            nresid = data.shape[0] % batch_size
            if nresid == 0:
                reshape_arg = [int(data.shape[0]/batch_size),batch_size]
                reshape_arg.extend(list(data.shape[1:]))
                L = [d for d in data.reshape(reshape_arg)]
            else:
                reshape_arg = [int(data.shape[0]/batch_size),batch_size]
                reshape_arg.extend(list(data.shape[1:]))
                L = [d for d in data[:-nresid].reshape(reshape_arg)]
                L.append(data[-nresid:])
                
            return L

        xbatches = get_minibatch_list(Xdata)
        ybatches = get_minibatch_list(Ydata)
        return xbatches, ybatches

    def get_batches(self,DEL_X_data,batch_size = None):
        """
        returns a list of batches to be processed.
        DEL_X_data a dictionary of the type {'DEL': train_DEL, 'X' : train_X}
        """
        X = DEL_X_data['X']
        DEL = DEL_X_data['DEL']
        return self._reshape_in_minibatches(X, DEL, batch_size)



class BECASMesh(object):
    def __init__(self,value_dict):
        """
        a class to load and manipulate BECAS related data
        from python.
        """
        self.el_2d = np.array(value_dict['el_2d'], dtype = np.integer)
        self.ne_2d = int(value_dict['ne_2d'])
        self.nl_2d = np.array(value_dict['nl_2d'])
        
    @staticmethod
    def getFieldArray():
        field_array = ['ne_2d', 'el_2d','nl_2d']
        return field_array

    @staticmethod
    def make_BECAS_mesh_array_from_wmbladeIO(matlab_blade_io):
        """
        a factory method to facilitate reading from matlab.

        Gets a matlab blade IO manager and uses it to extract the mesh nodes and connectivity for all cross sections

        Arguments:
          matlab_blade_io : a matlab blade IO object to be used for loading a particular wmblade file

        Outputs:
          an array of BECAS_mesh objects

        """
        def get_cell_array(field):
            """
            applies a cellfun to the utils cell array and returns a list of all the 
            fields of each object stored in the cell array
            """
            return wmb_io.engine_instance.eval("cellfun(@(x) x.%s , %s.Utils, 'UniformOutput',false)"%(field,matlab_blade_io.getBecasObjectName()), nargout = 1);

        field_array = BECASMesh.getFieldArray();
        becas_value_array = [ get_cell_array(k) for k in field_array]
        n_crossections = len(becas_value_array[0])

        constructor_inputs = [];
        for i in range(0, n_crossections):
            input_dict = {};
            for j,k in enumerate(field_array):
                input_dict[k] = becas_value_array[j][i]

            constructor_inputs.append(input_dict)

        return [BECASMesh(value_dict) for value_dict in constructor_inputs]
    
    @staticmethod
    def save_BECAS_mesh_array_to_folder(becas_mesh_array, folder):
        """
        A utility function to consistently save all the cross-sections of a wmblade
        to a folder.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

        for k , mesh in enumerate(becas_mesh_array):
            mesh.save_to_file("%s/%s"%(folder,'cross_section%i'%k))



    def plot_mesh(self, fig = None, linespec_string = '.-b'):
        """
        plots the mesh
        """

        node_inds = np.array(self.el_2d, dtype = np.integer)[:,1:5]
        node_num  = np.array(self.el_2d, dtype = np.integer)[:,0]
        n2darray = np.array(self.nl_2d)

        for el in range(0,self.ne_2d):
            el_inds = np.array(self.el_2d, dtype = np.integer)[el,1:5] # the indices of the nodes belonging to the element
            ind = self.el_2d[el,0] # the index of the element
            node_coords = n2darray[node_inds[el]-1][[0,1,2,3,0],1:]
            pplot.plot(node_coords[:,0],node_coords[:,1],linespec_string);

        
    @staticmethod
    def show_mesh_plot(nodes, elements, values, ax = False):

        y = nodes[:,0] 
        z = nodes[:,1] 

        def quatplot(y,z, quatrangles, values, ax=None, **kwargs):
            if not ax: ax=pplot.gca()
            yz = np.c_[y,z]
            verts= yz[quatrangles]
            pc = matplotlib.collections.PolyCollection(verts, **kwargs)
            pc.set_array(values)
            ax.add_collection(pc)
            ax.autoscale()
            return pc

        pc = quatplot(y,z, np.asarray(elements), values, ax = ax)


    def plot_mesh_value(self, val = None, fig = None):
        """
        plots a value on the mesh
        """
        node_inds = np.array(self.el_2d, dtype = np.integer)[:,1:5]
        BECASMesh.show_mesh_plot(self.nl_2d[:,1:],self.el_2d[:,1:5]-1,val)

    def save_to_file(self, filename):
        np.savez(filename, ne_2d = self.ne_2d, el_2d = self.el_2d, nl_2d = self.nl_2d )

    @staticmethod
    def make_BECAS_mesh_from_file(input_file):
        """
        the input file contains a numpy array 
        that contains the necessary inputs to the constructor.
        """
        value_dict = np.load(input_file)
        return BECASMesh(value_dict)


