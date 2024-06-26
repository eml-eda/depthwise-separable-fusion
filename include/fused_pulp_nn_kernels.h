#include "pmsis.h"

void __attribute__ ((noinline)) pulp_nn_matmul_ANY_CHW(
  const int8_t * pWeight,
  uint8_t *      pInBuffer,
  uint16_t       ch_out, // N.B This can be a subset of the total ch_out in Co_parallel kernels
  uint16_t       num_col_im2col,
  uint16_t       out_shift,
  uint16_t       out_mult,
  int32_t *      k,
  int32_t *      lambda,
  const int8_t * bias,
  uint8_t *      pOut,
  int            flag_relu,
  int            flag_batch_norm,
  int            dim_out_shift // Shift required to store the following channel (dim_out^2)
);
void __attribute__ ((noinline)) pulp_nn_pointwise_Co_parallel_CHW_CHW(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm,
  int             original_dim_im_in_x,
  int             original_dim_im_in_y,
  int             original_ch_in,
  int             original_dim_im_out_x,
  int             original_dim_im_out_y,
  int             original_ch_out
);

void __attribute__ ((noinline)) pulp_nn_pointwise_Co_parallel_HWC_CHW(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm,
  int             original_dim_im_in_x,
  int             original_dim_im_in_y,
  int             original_ch_in,
  int             original_dim_im_out_x,
  int             original_dim_im_out_y,
  int             original_ch_out
);


// Base kernels
void __attribute__ ((noinline)) pulp_nn_pointwise_HoWo_parallel_HWC_CHW(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer, // Not a fake buffer anymore, needs ch_in * 8 space.
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm,
  int             original_dim_im_in_x,
  int             original_dim_im_in_y,
  int             original_ch_im_in,
  int             original_dim_im_out_x,
  int             original_dim_im_out_y,
  int             original_ch_im_out
);
void __attribute__ ((noinline)) pulp_nn_pointwise_HoWo_parallel_CHW_CHW(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer, // Not a fake buffer anymore, needs ch_in * 8 space.
  const int8_t *  bias,
  uint8_t *       pOutBuffer,
  const int8_t *  pWeight,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm,
  int             original_dim_im_in_x,
  int             original_dim_im_in_y,
  int             original_ch_im_in,
  int             original_dim_im_out_x,
  int             original_dim_im_out_y,
  int             original_ch_im_out
);
void __attribute__ ((noinline)) pulp_nn_depthwise_generic_HWC_HWC(
                               const uint8_t * Im_in,
                               uint8_t *       bufferC,
                               const int8_t *  bias,
                               uint8_t *       Im_out,
                               const int8_t *  wt,
                               uint8_t *       bufferB,
                               int32_t *       k,
                               int32_t *       lambda,
                               uint16_t        out_mult,
                               uint16_t        out_shift,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          FLAG_RELU,
                               int8_t          FLAG_BATCH_NORM,
                               int             original_dim_im_in_x,
                               int             original_dim_im_in_y,
                               int             original_ch_im_in,
                               int             original_dim_im_out_x,
                               int             original_dim_im_out_y,
                               int             original_ch_im_out
  );
void __attribute__ ((noinline)) pulp_nn_depthwise_generic_CHW_HWC(
                               const uint8_t * Im_in,
                               uint8_t *       bufferC,
                               const int8_t *  bias,
                               uint8_t *       Im_out,
                               const int8_t *  wt,
                               uint8_t *       bufferB,
                               int32_t *       k,
                               int32_t *       lambda,
                               uint16_t        out_mult,
                               uint16_t        out_shift,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          FLAG_RELU,
                               int8_t          FLAG_BATCH_NORM,
                               int             original_dim_im_in_x,
                               int             original_dim_im_in_y,
                               int             original_ch_im_in,
                               int             original_dim_im_out_x,
                               int             original_dim_im_out_y,
                               int             original_ch_im_out
  );
void __attribute__ ((noinline)) pulp_nn_depthwise_generic_CHW_CHW(
                               const uint8_t * Im_in,
                               uint8_t *       bufferC,
                               const int8_t *  bias,
                               uint8_t *       Im_out,
                               const int8_t *  wt,
                               uint8_t *       bufferB,
                               int32_t *       k,
                               int32_t *       lambda,
                               uint16_t        out_mult,
                               uint16_t        out_shift,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          FLAG_RELU,
                               int8_t          FLAG_BATCH_NORM,
                               int             original_dim_im_in_x,
                               int             original_dim_im_in_y,
                               int             original_ch_im_in,
                               int             original_dim_im_out_x,
                               int             original_dim_im_out_y,
                               int             original_ch_im_out
  );

// Fused kernels
void __attribute__ ((noinline)) pulp_nn_dw_pw_CHW_CHW_HWC_rows_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, //  FD x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_dw_pw_HWC_HWC_HWC_rows_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, //  FD x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_pw_dw_CHW_CHW_CHW_channels_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, //  FD x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_CHW_channels_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, //  FD x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_CHW_rows_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, // FD x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_HWC_rows_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, // IB x CH_IN x W_OUT x H_OUT
                               const int8_t *  bias_pw,
                               const int8_t *  bias_dw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);
void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_HWC_channels_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, 
                               uint8_t *       intermediateBuffer, 
                               const int8_t *  bias_dw,
                               const int8_t *  bias_pw,
                               uint8_t *       Im_out,
                               const int8_t *  weights_dw,
                               const int8_t *  weights_pw,
                               int32_t *       k_dw,
                               int32_t *       k_pw,
                               int32_t *       lambda_dw,
                               int32_t *       lambda_pw,
                               uint16_t        out_mult_dw,
                               uint16_t        out_mult_pw,
                               uint16_t        out_shift_dw,
                               uint16_t        out_shift_pw,
                               const uint16_t  dim_im_in_x,
                               const uint16_t  dim_im_in_y,
                               const uint16_t  ch_im_in,
                               const uint16_t  dim_im_out_x,
                               const uint16_t  dim_im_out_y,
                               const uint16_t  ch_im_out,
                               const uint16_t  dim_kernel_x,
                               const uint16_t  dim_kernel_y,
                               const uint16_t  padding_y_top,
                               const uint16_t  padding_y_bottom,
                               const uint16_t  padding_x_left,
                               const uint16_t  padding_x_right,
                               const uint16_t  stride_x,
                               const uint16_t  stride_y,
                               int8_t          flag_relu_dw,
                               int8_t          flag_relu_pw,
                               int8_t          flag_batch_norm_dw,
                               int8_t          flag_batch_norm_pw,
                               uint16_t        fused_batch_size
  
);