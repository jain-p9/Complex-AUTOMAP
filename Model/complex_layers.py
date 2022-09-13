import tensorflow as tf
import tensorflow.keras.layers as layers

def complex_to_channels(image):            #output is of size (number of images, height, width*2)
    """Convert data from complex to channels."""
    image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
    shape_out = tf.concat(
        [tf.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out
    
class CDense(tf.keras.layers.Layer):

    def __init__(self, num_output_units):
      super(CDense, self).__init__()
      self.num_units_2 = num_output_units 
      self.num_units = num_output_units//2
      self.real_wt = layers.Dense(self.num_units) 
      self.imag_wt = layers.Dense(self.num_units) 
  
    def build(self, input_shape):
      self.built = True
    
    def call(self, input):
      in_channels = input.shape[-1] 

      in_real = input[:,:in_channels//2]
      in_imag = input[:,in_channels//2:]
      
      out_real_real = self.real_wt(in_real) 
      out_imag_imag = self.imag_wt(in_imag)
      out_real_imag = self.imag_wt(in_real)
      out_imag_real = self.real_wt(in_imag)
      
      out_real = out_real_real-out_imag_imag
      out_imag = out_real_imag+out_imag_real
      complex_output = tf.complex(out_real, out_imag)
     
      channels_output = complex_to_channels(complex_output)
     
      return channels_output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      shape[-1] = self.num_units*2
      return tf.TensorShape(shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
            'num_output_units': self.num_units_2,
        })
      return config



class CConv2D(tf.keras.Model):

    def __init__(self, num_outputs, kernel_size, padding=None, activity_regularizer=None):
      super(CConv2D, self).__init__()
      self.num_outputs_2 = num_outputs
      self.num_outputs = num_outputs // 2
      self.kernel_size = kernel_size
      self.padding = padding
      self.activity_regularizer = activity_regularizer
      self.conv_real = layers.Conv2D(self.num_outputs, self.kernel_size, padding=padding, activity_regularizer=activity_regularizer) #for real part of kernel
      self.conv_imag = layers.Conv2D(self.num_outputs, self.kernel_size, padding=padding, activity_regularizer=activity_regularizer) #for imag part of kernel

    def build(self, input_shape):
      self.built = True

    def call(self, input):
      in_channels = input.shape[-1] 

      in_real = input[:,:,:,:in_channels//2]
      in_imag = input[:,:,:,in_channels//2:]
    
      real_real = self.conv_real(in_real)
      real_imag = self.conv_imag(in_real)
      imag_real = self.conv_real(in_imag)
      imag_imag = self.conv_imag(in_imag)

      out_real = real_real-imag_imag
      out_imag = imag_real+real_imag

      complex_output = tf.complex(out_real, out_imag)

      channels_output = complex_to_channels(complex_output)

      self.out_h = channels_output.shape[1]
      self.out_w = channels_output.shape[2]
      
      return channels_output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      shape[1] = self.out_h
      shape[2] = self.out_w
      shape[-1] = self.num_outputs * 2
      return tf.TensorShape(shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
            'num_outputs': self.num_outputs_2,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'activity_regularizer': self.activity_regularizer,
        })
      return config

    

class CConv2DTranspose(tf.keras.layers.Layer):

    def __init__(self, num_outputs, kernel_size, padding=None):
      super(CConv2DTranspose, self).__init__()
      self.num_outputs_2 = num_outputs
      self.num_outputs = num_outputs // 2
      self.kernel_size = kernel_size
      self.padding = padding
      self.conv_real = layers.Conv2DTranspose(self.num_outputs, self.kernel_size, padding=padding) #for real part of kernel
      self.conv_imag = layers.Conv2DTranspose(self.num_outputs, self.kernel_size, padding=padding) #for imag part of kernel

    def build(self, input_shape):
      self.built = True

    def call(self, input):
      in_channels = input.shape[-1] 
   
      in_real = input[:,:,:,:in_channels//2]
      in_imag = input[:,:,:,in_channels//2:]
      
      real_real = self.conv_real(in_real)
      real_imag = self.conv_imag(in_real)
      imag_real = self.conv_real(in_imag)
      imag_imag = self.conv_imag(in_imag)
     
      out_real = real_real-imag_imag
      out_imag = imag_real+real_imag

      complex_output = tf.complex(out_real, out_imag)
      
      channels_output = complex_to_channels(complex_output)

      self.out_h = channels_output.shape[1]
      self.out_w = channels_output.shape[2]
     
      return channels_output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      shape[1] = self.out_h
      shape[2] = self.out_w
      shape[-1] = self.num_outputs * 2
      return tf.TensorShape(shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
            'num_outputs': self.num_outputs_2,
            'kernel_size': self.kernel_size,
            'padding': self.padding,
        })
      return config

    


class CReshape(tf.keras.layers.Layer):

    def __init__(self, im_h, im_w, out_ch):
      super(CReshape, self).__init__()
      self.im_h = im_h 
      self.im_w = im_w 
      self.out_ch = out_ch 

    def build(self, input_shape):
      self.built = True

    def call(self, input):
      in_len = input.shape[-1] 
   
      in_real = input[:,:in_len//2]
      in_imag = input[:,in_len//2:]
      
      out_real = tf.reshape(in_real, (-1,self.im_h,self.im_w))
      out_imag = tf.reshape(in_imag, (-1,self.im_h,self.im_w))
      
      out_img = tf.stack([out_real,out_imag],axis=-1)
      
      return out_img

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      out_shape[0] = shape[0]
      out_shape[1] = self.im_h
      out_shape[2] = self.im_w
      out_shape[3] = self.out_ch
      return tf.TensorShape(out_shape)

    def get_config(self):
      config = super().get_config().copy()
      config.update({
            'im_h': self.im_h,
            'im_w': self.im_w,
            'out_ch': self.out_ch,
        })
      return config
      
      
      
class CFlatten(tf.keras.layers.Layer):

    def __init__(self):
      super(CFlatten, self).__init__()

    def build(self, input_shape):
      self.in_c = input_shape[3]
      self.in_w = input_shape[2]
      self.in_h = input_shape[1]
      self.built = True

    def call(self, input):
      real = input[:,:,:,:self.in_c//2]
      imag = input[:,:,:,self.in_c//2:]

      real_flat = tf.reshape(real, (-1,self.in_h*self.in_w*(self.in_c//2)))
      imag_flat = tf.reshape(imag, (-1,self.in_h*self.in_w*(self.in_c//2)))

      output = tf.concat([real_flat,imag_flat],-1)
      
      return output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      out_shape[0] = shape[0]
      out_shape[1] = self.in_c*self.in_h*self.in_w
      return tf.TensorShape(out_shape)


class abs_layer(tf.keras.layers.Layer):

    def __init__(self):
      super(abs_layer, self).__init__()

    def build(self, input_shape):
      self.in_c = input_shape[3]
      self.built = True

    def call(self, input):
      real_part = input[:,:,:,:self.in_c//2]
      imag_part = input[:,:,:,self.in_c//2:]

      output = tf.math.sqrt(real_part*real_part+imag_part*imag_part)
      
      return output

    def compute_output_shape(self, input_shape):
      shape = tf.TensorShape(input_shape).as_list()
      shape[-1] = self.in_c//2
      return tf.TensorShape(shape)
