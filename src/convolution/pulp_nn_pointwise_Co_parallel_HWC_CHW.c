#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "fused_pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

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
) {
  int core_id = pi_core_id();

  // local vars
  int i_out_y, i_out_x, i_ker_y, i_ker_x;
  int Log2Core = log2(NUM_CORES);

  int chunk = (ch_out >> Log2Core) + ((ch_out & (NUM_CORES - 1)) != 0);

  /* defining the specific channels computed by each core */
  int start_channel, stop_channel;
  start_channel = min(chunk * core_id, ch_out);
  stop_channel = min(start_channel + chunk, ch_out);

  int eff_chunk = stop_channel - start_channel;

  int8_t *pW = pWeight + (start_channel * ch_in);

  int32_t *k0 = k + start_channel;
  int32_t *lambda0 = lambda + start_channel;

  if(eff_chunk)
  {
    for (i_out_y = 0; i_out_y < dim_out_y; i_out_y++)
    {
      i_out_x = 0;

      uint8_t *pOut = pOutBuffer + start_channel*(original_dim_im_out_x * original_dim_im_out_y) + (i_out_y * original_dim_im_out_x);

      for (int n = 0; n < (dim_out_x >> 1); n++)
      {
          uint8_t *pB = (pInBuffer + (i_out_x * original_ch_in) + (i_out_y * original_dim_im_in_x * original_ch_in));
          pulp_nn_matmul_ANY_CHW(
            pW,
            pB,
            eff_chunk,
            ch_in,
            out_shift,
            out_mult,
            k0,
            lambda0,
            bias,
            pOut,
            flag_relu,
            flag_batch_norm,
            (original_dim_im_out_x * original_dim_im_out_y)
          );
          i_out_x+=2;
          pOut+=2;
      }
      /* check if there is left-over for compute */
      if (i_out_x != dim_out_x)
      {
        const int8_t *pA = pWeight;

        for (int i = start_channel; i < stop_channel; i++)
        {
          int sum = 0;
          uint16_t dim_out = (original_dim_im_out_x * original_dim_im_out_y) * i;


          if (bias != NULL)
          {
            sum = ((int)(bias[i]));
          }

          uint8_t *pB = (pInBuffer + (i_out_x * original_ch_in) + (i_out_y * original_dim_im_in_x * original_ch_in));
          /* basically each time it process 4 entries */
          uint16_t  col_cnt_im2col = ch_in >> 2;

          for (int j=0 ; j < col_cnt_im2col; j++)
          {
            v4s inA = *((v4s*) pA);
            v4u inB = *((v4u*) pB);

            sum = SumDotp(inB, inA, sum);
            pA+=4;
            pB+=4;
          }
          col_cnt_im2col = ch_in & 0x3;
          while (col_cnt_im2col)
          {
            int8_t      inA1 = *pA++;
            uint8_t     inB1 = *pB++;
            asm volatile("": : :"memory");
            sum += inA1 * inB1;

            col_cnt_im2col--;
          }
          /* if activation layer follows batch normalization */
          if (flag_batch_norm && flag_relu)
          {
            *(pOut + dim_out) = pulp_nn_bn_quant_u8(sum, *k0, *lambda0, out_shift);
            k0++;
            lambda0++;
            //pOut++;
          }
          else
          {
            /* if there isn't batch normalization but there is activation layer */
            if(flag_relu == 1)
            {
              *(pOut + dim_out) = pulp_nn_quant_u8(sum, out_mult, out_shift);
            }
            else
            {
              *(pOut + dim_out) = (uint8_t) clip8(sum >> out_shift);
            }
          }
        }
            pOut++;
      }
    }
  }
  pi_cl_team_barrier(0);
}