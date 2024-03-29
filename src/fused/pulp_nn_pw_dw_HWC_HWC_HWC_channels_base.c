#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"
#include "fused_pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define clip8(x) __builtin_pulp_clipu_r(x, 255)
#if DEBUG>0
#define PRINT_VECTOR(x,dim) \
  for(int ijkl = 0; ijkl < dim; ijkl+=1) { \
    printf("%d ",x[ijkl]); \
  } \
  printf("\n");
#else
#define PRINT_VECTOR(x,dim)
#endif

void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_HWC_channels_base(
                               const uint8_t * Im_in,
                               uint8_t *       Im2ColBuffer, // Im2Col [8x KS x KS]
                               uint8_t *       intermediateBuffer, // Buffer [RIGHE_PW * CH_IN * H_IN * W_IN]
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
  ){
  
  int row = 0;
  int core_id = pi_core_id();
  uint8_t *pIn = Im_in;
  uint8_t *pOut = Im_out;

  uint16_t CHANNELS_BATCH_DW = fused_batch_size; 

  int channels_done = 0;
  for(; channels_done <= (ch_im_out - CHANNELS_BATCH_DW); channels_done += CHANNELS_BATCH_DW) {
    pulp_nn_pointwise_HoWo_parallel(
          /*pInBuffer=*/pIn,
          /*pIm2ColBuffer=*/NULL, 
          /*bias=*/bias_pw,
          /*pOutBuffer=*/intermediateBuffer,
          /*pWeight=*/weights_pw,
          /*k=*/k_pw,
          /*lambda=*/lambda_pw,
          /*out_mult=*/out_mult_pw,
          /*out_shift=*/out_shift_pw,
          /*dim_in_x=*/dim_im_in_x,
          /*dim_in_y=*/dim_im_in_y,
          /*ch_in=*/ch_im_in,
          /*dim_out_x=*/dim_im_in_x,
          /*dim_out_y=*/dim_im_in_y,
          /*ch_out=*/CHANNELS_BATCH_DW,
          /*dim_kernel_x=*/1,
          /*dim_kernel_y=*/1,
          /*padding_y_top=*/0,
          /*padding_y_bottom=*/0,
          /*padding_x_left=*/0,
          /*padding_x_right=*/0,
          /*stride_x=*/1,
          /*stride_y=*/1,
          /*flag_relu=*/flag_relu_pw,
          /*flag_batch_norm=*/flag_batch_norm_pw
          //dim_im_in_x,
          //dim_im_in_y,
          //ch_im_in,
          //dim_im_in_x,
          //dim_im_in_y,
          //ch_im_out

    );
    pulp_nn_depthwise_generic_HWC_HWC(
          /*Im_in=*/intermediateBuffer,
          /*bufferC=*/Im2ColBuffer,
          /*bias=*/NULL,
          /*Im_out=*/pOut,
          /*wt=*/weights_dw,
          /*bufferB=*/NULL,
          /*k=*/k_dw,
          /*lambda=*/lambda_dw,
          /*out_mult=*/out_mult_dw,
          /*out_shift=*/out_shift_dw,
          /*dim_im_in_x=*/dim_im_in_x,
          /*dim_im_in_y=*/dim_im_in_y,
          /*ch_im_in=*/CHANNELS_BATCH_DW,
          /*dim_im_out_x=*/dim_im_out_x,
          /*dim_im_out_y=*/dim_im_out_y,
          /*ch_im_out=*/CHANNELS_BATCH_DW,
          /*dim_kernel_x=*/dim_kernel_x,
          /*dim_kernel_y=*/dim_kernel_y,
          /*padding_y_top=*/padding_y_top,
          /*padding_y_bottom=*/padding_y_bottom,
          /*padding_x_left=*/padding_x_left,
          /*padding_x_right=*/padding_x_right,
          /*stride_x=*/stride_x,
          /*stride_y=*/stride_y,
          /*FLAG_RELU=*/flag_relu_dw,
          /*FLAG_BATCH_NORM=*/flag_batch_norm_dw,
          dim_im_in_x,
          dim_im_in_y,
          ch_im_out,
          dim_im_out_x,
          dim_im_out_y,
          ch_im_out
    );

    weights_pw += (1 * 1 * ch_im_in) * (CHANNELS_BATCH_DW);
    weights_dw += (dim_kernel_x * dim_kernel_y) * (CHANNELS_BATCH_DW);
    k_pw += (CHANNELS_BATCH_DW) * (flag_batch_norm_pw);
    lambda_pw += (CHANNELS_BATCH_DW) * (flag_batch_norm_pw);
    bias_pw += (CHANNELS_BATCH_DW) * (bias_pw!=NULL);
    pOut += CHANNELS_BATCH_DW;
  }

  int leftover = ch_im_out % CHANNELS_BATCH_DW;
  if(leftover) {
  #else
    pulp_nn_pointwise_HoWo_parallel(
          /*pInBuffer=*/pIn,
          /*pIm2ColBuffer=*/Im2ColBuffer,
          /*bias=*/NULL,
          /*pOutBuffer=*/intermediateBuffer,
          /*pWeight=*/weights_pw,
          /*k=*/k_pw,
          /*lambda=*/lambda_pw,
          /*out_mult=*/out_mult_pw,
          /*out_shift=*/out_shift_pw,
          /*dim_in_x=*/dim_im_in_x,
          /*dim_in_y=*/dim_im_in_y,
          /*ch_in=*/ch_im_in,
          /*dim_out_x=*/dim_im_in_x,
          /*dim_out_y=*/dim_im_in_y,
          /*ch_out=*/leftover,
          /*dim_kernel_x=*/1,
          /*dim_kernel_y=*/1,
          /*padding_y_top=*/0,
          /*padding_y_bottom=*/0,
          /*padding_x_left=*/0,
          /*padding_x_right=*/0,
          /*stride_x=*/1,
          /*stride_y=*/1,
          /*flag_relu=*/flag_relu_pw,
          /*flag_batch_norm=*/flag_batch_norm_pw
          //dim_im_in_x,
          //dim_im_in_y,
          //ch_im_in,
          //dim_im_in_x,
          //dim_im_in_y,
          //ch_im_out
    );

    pulp_nn_depthwise_generic_HWC_HWC(
          /*Im_in=*/intermediateBuffer,
          /*bufferC=*/Im2ColBuffer,
          /*bias=*/NULL,
          /*Im_out=*/pOut,
          /*wt=*/weights_dw,
          /*bufferB=*/NULL,
          /*k=*/k_dw,
          /*lambda=*/lambda_dw,
          /*out_mult=*/out_mult_dw,
          /*out_shift=*/out_shift_dw,
          /*dim_im_in_x=*/dim_im_in_x,
          /*dim_im_in_y=*/dim_im_in_y,
          /*ch_im_in=*/leftover,
          /*dim_im_out_x=*/dim_im_out_x,
          /*dim_im_out_y=*/dim_im_out_y,
          /*ch_im_out=*/leftover,
          /*dim_kernel_x=*/dim_kernel_x,
          /*dim_kernel_y=*/dim_kernel_y,
          /*padding_y_top=*/padding_y_top,
          /*padding_y_bottom=*/padding_y_bottom,
          /*padding_x_left=*/padding_x_left,
          /*padding_x_right=*/padding_x_right,
          /*stride_x=*/stride_x,
          /*stride_y=*/stride_y,
          /*FLAG_RELU=*/flag_relu_dw,
          /*FLAG_BATCH_NORM=*/flag_batch_norm_dw,
          dim_im_in_x,
          dim_im_in_y,
          ch_im_out,
          dim_im_out_x,
          dim_im_out_y,
          ch_im_out
    );
  }


  pi_cl_team_barrier();
}