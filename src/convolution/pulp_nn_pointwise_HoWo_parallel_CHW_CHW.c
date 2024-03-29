#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"
#include "fused_pulp_nn_kernels.h"

#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))
#define SumDotp(a, b, c)        __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x)                __builtin_pulp_clipu_r(x, 255)

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
) {
   int core_id = pi_core_id();

  // local vars
  int i_out_y, i_out_x;
  int Log2Core = log2(NUM_CORES);


  uint8_t extra_chunk = ((dim_out_y & (NUM_CORES-1)) != 0);
  uint8_t extra_chunk_r;
  uint16_t dim_out_x_r;
  uint8_t section;
  int core_id_r;

  if(extra_chunk && dim_out_x > 1)
  {
    Log2Core = log2(NUM_CORES >> 1);
    core_id_r = (core_id >> 1);
    dim_out_x_r = (dim_out_x >> 1);
    section = (core_id & 0x1);
    extra_chunk_r = ((dim_out_y & ((NUM_CORES >> 1) - 1)) != 0);
  }
  else
  {
    Log2Core = log2(NUM_CORES);
    core_id_r = core_id;
    dim_out_x_r = dim_out_x;
    section = 0;
    extra_chunk_r = extra_chunk;
    extra_chunk = 0;
  }

  uint8_t flag_dim_out_x_odd = dim_out_x & 0x0001;

  int chunk = (dim_out_y >> Log2Core) + extra_chunk_r;

  int start_pixel = min((chunk * core_id_r), dim_out_y);
  int stop_pixel = min(start_pixel + chunk, dim_out_y);

  uint8_t *pOut = pOutBuffer + (start_pixel * dim_out_x) + (section * dim_out_x_r);
  uint8_t *pIm2Col = pIm2ColBuffer + (core_id * ch_in * 2);

  for (i_out_y = start_pixel; i_out_y < stop_pixel; i_out_y++)
  {
    i_out_x= (section * dim_out_x_r);

    for(int n = 0; n<((dim_out_x_r + (section * flag_dim_out_x_odd)) >> 1); n++)
    {
      uint8_t *pB = pInBuffer + (i_out_x) + (i_out_y * dim_in_x);
      for(int cin = 0; cin < ch_in; cin++) {
        pIm2Col[cin] = (*pB);
        pIm2Col[cin + ch_in] = (*(pB + 1));
        asm volatile("": : :"memory");
        pB += (original_dim_im_in_y * original_dim_im_in_x);
      }
      pulp_nn_matmul_ANY_CHW(
        pWeight,
        pIm2Col,
        ch_out,
        ch_in,
        out_shift,
        out_mult,
        k,
        lambda,
        bias,
        pOut,
        flag_relu,
        flag_batch_norm,
        original_dim_im_out_x * original_dim_im_out_y
      );
      asm volatile("": : :"memory");
      i_out_x+=2;
      pOut+=2;
    }
    /* check if there is left-over for compute */
    if (((dim_out_x_r + (section * flag_dim_out_x_odd)) & 0x0001))
    {
      const int8_t *pA = pWeight;
      int32_t *k1 = k;
      int32_t *lambda1 = lambda;

      uint8_t *pIn = pInBuffer + i_out_x + (i_out_y * dim_in_x); // TODO: This may be broken
      // Im2Col - Easy since there is no padding.
      for(int cin = 0; cin<ch_in; cin++) {
        pIm2Col[cin] = (*pIn);
        pIn+= (original_dim_im_in_x * original_dim_im_in_y);
      }

      for (int i = 0; i < ch_out; i++)
      {
        int sum = 0;
        uint16_t dim_out = (original_dim_im_out_x * original_dim_im_out_y) * i;

        if (bias != NULL)
        {
          sum = ((int)(bias[i]));
        }

        uint8_t *pB = pIm2Col;
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
          *(pOut + dim_out) = pulp_nn_bn_quant_u8(sum, *k1, *lambda1, out_shift);
          k1++;
          lambda1++;
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
      pOut+=1;
    }
    pOut+=(extra_chunk * ((dim_out_x_r + ((1 - section) * flag_dim_out_x_odd)))); // This may be broken too.
  }
  pi_cl_team_barrier(0);
}
