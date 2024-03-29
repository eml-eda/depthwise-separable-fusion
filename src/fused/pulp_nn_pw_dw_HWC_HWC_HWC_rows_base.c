#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"
#include "fused_pulp_nn_kernels.h"
#include "string.h"

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

// Richiede shift piu' complicati e salti nell'array della depthwise.
void __attribute__ ((noinline)) pulp_nn_pw_dw_HWC_HWC_HWC_rows_base(
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
   // General PULP variables
  int core_id = pi_core_id();

  // Fused kernel specific vars
  uint16_t intermediate_rows = fused_batch_size;
  
  // Additional pointers
  uint8_t *pIn = Im_in;
  uint8_t *pOut = Im_out;

  // New variables to handle the slide in the image
  uint16_t padding_bottom;
  uint16_t dim_im_out_y_no_pad = dim_im_out_y - padding_y_bottom; // TODO: This works only for padding <=1 
  uint16_t output_rows; // Different from DW + PW
  uint16_t output_rows_padded;

  if(intermediate_rows == dim_im_in_y) {
    // Not an actual fused kernel
    padding_bottom = padding_y_bottom;
    output_rows = dim_im_out_y;
    output_rows_padded = output_rows;
  } else {
    padding_bottom = 0;
    output_rows = (intermediate_rows - (dim_kernel_y - 1) - 1)/stride_y + 1;
    output_rows_padded = output_rows + padding_y_top/stride_y;
  }
  // Righe nuove su cui calcolare la PW, le altre vengono riutilizzate.
  uint16_t intermediate_rows_shift_in = output_rows * stride_y;
  // Se la depthwise aveva padding, lo shift va ridotto
  uint16_t intermediate_rows_shift_in_padded = intermediate_rows_shift_in;// - padding_y_top;
  uint16_t current_in_row = 0;


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
        /*dim_in_y=*/intermediate_rows,
        /*ch_in=*/ch_im_in,
        /*dim_out_x=*/dim_im_in_x,
        /*dim_out_y=*/intermediate_rows,
        /*ch_out=*/ch_im_out,
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
        /*dim_im_in_y=*/intermediate_rows,
        /*ch_im_in=*/ch_im_out,
        /*dim_im_out_x=*/dim_im_out_x,
        /*dim_im_out_y=*/output_rows_padded,
        /*ch_im_out=*/ch_im_out,
        /*dim_kernel_x=*/dim_kernel_x,
        /*dim_kernel_y=*/dim_kernel_y,
        /*padding_y_top=*/padding_y_top,
        /*padding_y_bottom=*/padding_bottom,
        /*padding_x_left=*/padding_x_left,
        /*padding_x_right=*/padding_x_right,
        /*stride_x=*/stride_x,
        /*stride_y=*/stride_y,
        /*FLAG_RELU=*/flag_relu_dw,
        /*FLAG_BATCH_NORM=*/flag_batch_norm_dw,
        dim_im_in_x,
        intermediate_rows,
        ch_im_out,
        dim_im_out_x,
        dim_im_out_y,
        ch_im_out
  );

  pIn += intermediate_rows * dim_im_in_x * ch_im_in; 
  pOut+= output_rows_padded * dim_im_out_x * ch_im_out;
  if(output_rows< dim_im_out_y && core_id == 0) {
      memmove(intermediateBuffer, intermediateBuffer + (intermediate_rows_shift_in_padded * dim_im_in_x * ch_im_out), intermediate_rows * dim_im_in_x * ch_im_out); 
  }
  pi_cl_team_barrier(0);
  current_in_row+= intermediate_rows;

  int current_out_row;
  for(current_out_row = output_rows_padded; current_out_row<= (dim_im_out_y_no_pad - output_rows); current_out_row+= output_rows) {
  pulp_nn_pointwise_HoWo_parallel(
        /*pInBuffer=*/pIn,
        /*pIm2ColBuffer=*/NULL,
        /*bias=*/bias_pw,
        /*pOutBuffer=*/intermediateBuffer + (intermediate_rows - intermediate_rows_shift_in_padded) * dim_im_in_x * ch_im_out,
        /*pWeight=*/weights_pw,
        /*k=*/k_pw,
        /*lambda=*/lambda_pw,
        /*out_mult=*/out_mult_pw,
        /*out_shift=*/out_shift_pw,
        /*dim_in_x=*/dim_im_in_x,
        /*dim_in_y=*/intermediate_rows_shift_in_padded,
        /*ch_in=*/ch_im_in,
        /*dim_out_x=*/dim_im_in_x,
        /*dim_out_y=*/intermediate_rows_shift_in_padded,
        /*ch_out=*/ch_im_out,
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
  );

  pulp_nn_depthwise_generic_HWC_HWC(
        /*Im_in=*/intermediateBuffer,
        /*bufferC=*/Im2ColBuffer,
        /*bias=*/bias_dw,
        /*Im_out=*/pOut,
        /*wt=*/weights_dw,
        /*bufferB=*/NULL,
        /*k=*/k_dw,
        /*lambda=*/lambda_dw,
        /*out_mult=*/out_mult_dw,
        /*out_shift=*/out_shift_dw,
        /*dim_im_in_x=*/dim_im_in_x,
        /*dim_im_in_y=*/intermediate_rows,
        /*ch_im_in=*/ch_im_out,
        /*dim_im_out_x=*/dim_im_out_x,
        /*dim_im_out_y=*/output_rows,
        /*ch_im_out=*/ch_im_out,
        /*dim_kernel_x=*/dim_kernel_x,
        /*dim_kernel_y=*/dim_kernel_y,
        /*padding_y_top=*/0,
        /*padding_y_bottom=*/0,
        /*padding_x_left=*/padding_x_left,
        /*padding_x_right=*/padding_x_right,
        /*stride_x=*/stride_x,
        /*stride_y=*/stride_y,
        /*FLAG_RELU=*/flag_relu_dw,
        /*FLAG_BATCH_NORM=*/flag_batch_norm_dw,
        dim_im_in_x,
        intermediate_rows,
        ch_im_out,
        dim_im_out_x,
        dim_im_out_y,
        ch_im_out
  );
  pIn += intermediate_rows_shift_in_padded * dim_im_in_x * ch_im_in; 
  pOut+= output_rows* dim_im_out_x * ch_im_out;

  if(core_id == 0)
    memmove(intermediateBuffer, intermediateBuffer + (intermediate_rows_shift_in* dim_im_in_x * ch_im_out), intermediate_rows * dim_im_in_x * ch_im_out); 
  pi_cl_team_barrier(0);

  current_in_row+= intermediate_rows_shift_in_padded;
  // Only iteration 0 needs special treatment
  intermediate_rows_shift_in_padded = intermediate_rows_shift_in;

  }
  // MIDDLE FUSED BLOCK END ( NO PADDING TOP/ BOTTOM)


  // LEFTOVER + BOTTOM PADDING
  if(current_out_row<dim_im_out_y) {
    int leftover_output_rows = dim_im_out_y - current_out_row;
    int leftover_input_rows = dim_im_in_y - current_in_row;
    //int leftover_rows_shift_in = ;
    int leftover_intermediate_rows = (leftover_input_rows + (intermediate_rows - intermediate_rows_shift_in));
  pulp_nn_pointwise_HoWo_parallel(
      /*pInBuffer=*/pIn,
      /*pIm2ColBuffer=*/NULL,
      /*bias=*/bias_pw,
      /*pOutBuffer=*/intermediateBuffer + (intermediate_rows - intermediate_rows_shift_in_padded) * dim_im_in_x * ch_im_out,
      /*pWeight=*/weights_pw,
      /*k=*/k_pw,
      /*lambda=*/lambda_pw,
      /*out_mult=*/out_mult_pw,
      /*out_shift=*/out_shift_pw,
      /*dim_in_x=*/dim_im_in_x,
      /*dim_in_y=*/leftover_input_rows,
      /*ch_in=*/ch_im_in,
      /*dim_out_x=*/dim_im_in_x,
      /*dim_out_y=*/leftover_input_rows,
      /*ch_out=*/ch_im_out,
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
        /*dim_im_in_y=*/leftover_intermediate_rows,
        /*ch_im_in=*/ch_im_out,
        /*dim_im_out_x=*/dim_im_out_x,
        /*dim_im_out_y=*/leftover_output_rows,
        /*ch_im_out=*/ch_im_out,
        /*dim_kernel_x=*/dim_kernel_x,
        /*dim_kernel_y=*/dim_kernel_y,
        /*padding_y_top=*/0,
        /*padding_y_bottom=*/padding_y_bottom,
        /*padding_x_left=*/padding_x_left,
        /*padding_x_right=*/padding_x_right,
        /*stride_x=*/stride_x,
        /*stride_y=*/stride_y,
        /*FLAG_RELU=*/flag_relu_dw,
        /*FLAG_BATCH_NORM=*/flag_batch_norm_dw,
        dim_im_in_x,
        intermediate_rows,
        ch_im_out,
        dim_im_out_x,
        dim_im_out_y,
        ch_im_out
  );

  }

      


  pi_cl_team_barrier();
}