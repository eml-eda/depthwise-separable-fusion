#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"
#include "fused_pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define clip8(x) __builtin_pulp_clipu_r(x, 255)


void __attribute__ ((noinline)) pulp_nn_dw_pw_CHW_CHW_HWC_rows_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // 8xKS x KS
                               uint8_t *       intermediateBuffer, // RIGHE_DW_OUT x CH_IN x W_OUT x H_OUT
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
  ){

  // General PULP variables
  int core_id = pi_core_id();

  // Fused kernel specific vars
  uint16_t intermediate_rows = fused_batch_size;
  
  // Additional pointers
  uint8_t *pIn = Im_in;
  uint8_t *pOut = Im_out;

  // New variables to handle the slide in the image
  uint16_t padding_bottom;
  uint16_t input_rows;
  uint16_t dim_im_out_y_no_pad = dim_im_out_y - padding_y_bottom; // TODO: This works only for padding <=1 
  uint16_t input_rows_padded;
  if(intermediate_rows==dim_im_out_y) {
    padding_bottom = padding_y_bottom;
    input_rows = dim_im_in_y;
    input_rows_padded = dim_im_in_y;
  } else {
    padding_bottom = 0;
    input_rows = (intermediate_rows -1) * stride_y + dim_kernel_y;
    input_rows_padded = input_rows - padding_y_top;
  }
  int current_in_row = 0;

  // CONV BLOCK START 
  pulp_nn_depthwise_generic_CHW_HWC(
    pIn,
    Im2ColBuffer, 
    bias_dw, 
    intermediateBuffer, 
    weights_dw, 
    (void *) 0,
    k_dw, 
    lambda_dw, 
    out_mult_dw, 
    out_shift_dw, 
    dim_im_in_x, 
    input_rows_padded,  // New
    ch_im_in,
    dim_im_out_x, 
    intermediate_rows,  // New
    ch_im_in, 
    dim_kernel_x, 
    dim_kernel_y, 
    padding_y_top, 
    padding_bottom, 
    padding_x_left, 
    padding_x_right, 
    stride_x, 
    stride_y, 
    flag_relu_dw, 
    flag_batch_norm_dw, 
    dim_im_in_x,
    dim_im_in_y,
    ch_im_in,
    dim_im_out_x,
    intermediate_rows,
    ch_im_in
  );
  pulp_nn_pointwise_HoWo_parallel_HWC_CHW(
                                          /*pInBuffer=*/  intermediateBuffer, 
                                          /*pIm2ColBuffer=*/NULL, 
                                          /*bias=*/bias_pw, 
                                          /*pOutBuffer*/pOut, 
                                          /*pWeight*/weights_pw, 
                                          /*k*/k_pw, 
                                          /*lambda*/lambda_pw, 
                                          /*out_mult*/out_mult_pw, 
                                          /*out_shift*/out_shift_pw,
                                          /*dim_in_x*/dim_im_out_x, 
                                          /*dim_in_y*/intermediate_rows, 
                                          /*ch_in*/ch_im_in, 
                                          /*dim_out_x*/dim_im_out_x,
                                          /*dim_out_y*/intermediate_rows, 
                                          /*ch_out*/ch_im_out, 
                                          /*dim_kernel_x*/1, 
                                          /*dim_kernel_y*/1, 
                                          /*padding*/0, 
                                          /*padding*/0, 
                                          /*padding*/0, 
                                          /*padding*/0, 
                                          /*stride*/1, 
                                          /*stride*/1, 
                                          /*flag_relu*/flag_relu_pw, 
                                          /*bn*/flag_batch_norm_pw, 
                                          dim_im_out_x,
                                          dim_im_out_y,
                                          ch_im_in,
                                          dim_im_out_x,
                                          dim_im_out_y,
                                          ch_im_out);

  // CONV BLOCK END

  // SHIFT INTERMEDIATE BUFFER START
  pOut += intermediate_rows* dim_im_out_x;
  pIn += (stride_y * intermediate_rows - padding_y_top) * dim_im_in_x; 
  current_in_row+= intermediate_rows * stride_y - padding_y_top;
  // SHIFT INTERMEDIATE BUFFER END

  // MIDDLE FUSED BLOCK START ( NO PADDING TOP/ BOTTOM)
  int current_out_row;
  for(current_out_row = intermediate_rows; current_out_row <= (dim_im_out_y_no_pad - intermediate_rows); current_out_row+= intermediate_rows) {
    pulp_nn_depthwise_generic_CHW_HWC(pIn,
                                      Im2ColBuffer, 
                                      bias_dw, 
                                      intermediateBuffer, 
                                      weights_dw, 
                                      (void *) 0, 
                                      k_dw,
                                      lambda_dw, 
                                      out_mult_dw, 
                                      out_shift_dw, 
                                      dim_im_in_x,
                                      input_rows, 
                                      ch_im_in,
                                      dim_im_out_x, 
                                      intermediate_rows, 
                                      ch_im_in, 
                                      dim_kernel_x, 
                                      dim_kernel_y, 
                                      0, 
                                      0, 
                                      padding_x_left, 
                                      padding_x_right, 
                                      stride_x, 
                                      stride_y, 
                                      flag_relu_dw, 
                                      flag_batch_norm_dw, 
                                      dim_im_in_x,
                                      dim_im_in_y,
                                      ch_im_in,
                                      dim_im_out_x, 
                                      dim_im_out_y,
                                      ch_im_in
                                      );
    
    pulp_nn_pointwise_HoWo_parallel_HWC_CHW(
                                            intermediateBuffer, 
                                            NULL, 
                                            bias_pw, 
                                            pOut, 
                                            weights_pw, 
                                            k_pw, 
                                            lambda_pw, 
                                            out_mult_pw, 
                                            out_shift_pw,
                                            dim_im_out_x, 
                                            intermediate_rows, 
                                            ch_im_in, 
                                            dim_im_out_x, 
                                            intermediate_rows, 
                                            ch_im_out, 
                                            1, 
                                            1, 
                                            0, 
                                            0, 
                                            0, 
                                            0, 
                                            1, 
                                            1, 
                                            flag_relu_pw, 
                                            flag_batch_norm_pw, 
                                            dim_im_out_x,
                                            dim_im_out_y,
                                            ch_im_in,
                                            dim_im_out_x,
                                            dim_im_out_y,
                                            ch_im_out);
  #endif

    pOut += intermediate_rows * dim_im_out_x;
    pIn += (intermediate_rows * stride_y)  * dim_im_in_x;
    current_in_row+= intermediate_rows * stride_y;
  }
  // MIDDLE FUSED BLOCK END ( NO PADDING TOP/ BOTTOM)

  // LEFTOVER + BOTTOM PADDING
  if(current_out_row<dim_im_out_y) {
    int leftover_output_rows = dim_im_out_y - current_out_row;
    int leftover_input_rows = dim_im_in_y - current_in_row;
    int leftover_input_rows_padded = leftover_input_rows - padding_y_bottom;
    // pIN and pOut already setup from before
    pulp_nn_depthwise_generic_CHW_HWC(pIn, 
                                      Im2ColBuffer, 
                                      bias_dw, 
                                      intermediateBuffer, 
                                      weights_dw, 
                                      NULL, 
                                      k_dw, 
                                      lambda_dw, 
                                      out_mult_dw, 
                                      out_shift_dw, 
                                      dim_im_in_x, 
                                      leftover_input_rows, 
                                      ch_im_in,
                                      dim_im_out_x, 
                                      leftover_output_rows, 
                                      ch_im_in, 
                                      dim_kernel_x, 
                                      dim_kernel_y, 
                                      0, 
                                      padding_y_bottom, 
                                      padding_x_left, 
                                      padding_x_right, 
                                      stride_x, 
                                      stride_y, 
                                      flag_relu_dw, 
                                      flag_batch_norm_dw, 
                                      dim_im_in_x,
                                      dim_im_in_y,
                                      ch_im_in,
                                      dim_im_out_x,
                                      dim_im_out_y,
                                      ch_im_in
                                      );
    pulp_nn_pointwise_HoWo_parallel_HWC_CHW(
                                            intermediateBuffer,
                                            NULL, 
                                            bias_pw, 
                                            pOut, 
                                            weights_pw, 
                                            k_pw, 
                                            lambda_pw, 
                                            out_mult_pw, 
                                            out_shift_pw,
                                            dim_im_out_x, 
                                            leftover_output_rows, 
                                            ch_im_in, 
                                            dim_im_out_x, 
                                            leftover_output_rows, 
                                            ch_im_out, 
                                            1, 
                                            1, 
                                            0, 
                                            0, 
                                            0, 
                                            0, 
                                            1, 
                                            1, 
                                            flag_relu_pw, 
                                            flag_batch_norm_pw, 
                                            dim_im_out_x,
                                            dim_im_out_y,
                                            ch_im_in,
                                            dim_im_out_x,
                                            dim_im_out_y,
                                            ch_im_out);
  }

}