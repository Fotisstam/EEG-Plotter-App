/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"
#include "arm_math.h"
#include <math.h>

/* Private defines -----------------------------------------------------------*/

#define PROTO_SYNC0          0xAAU
#define PROTO_SYNC1          0x55U

#define NUM_CHANNELS         32U
#define DISPLAY_BINS         128U
#define FFT_LENGTH           256U

#define PROTO_HDR_SIZE       7U
#define PROTO_DATA_BYTES     (NUM_CHANNELS * DISPLAY_BINS * 2U)
#define PROTO_FRAME_SIZE     (PROTO_HDR_SIZE + PROTO_DATA_BYTES + 2U)

#define FRAME_PERIOD_MS      20U
#define SAMPLE_RATE          256U

#define STREAM_MODE_FFT      'F'
#define STREAM_MODE_RAW      'R'
#define STREAM_MODE_SENSORS  'S'
#define STREAM_MODE_OSC      'O'

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Private variables ---------------------------------------------------------*/

static uint16_t g_seq = 0U;
static uint32_t g_raw_sample_index = 0U;

static uint8_t frame[2][PROTO_FRAME_SIZE];
static arm_rfft_fast_instance_f32 fft_instance;
static float fft_input[FFT_LENGTH];
static float fft_output[FFT_LENGTH];

/* Private function prototypes -----------------------------------------------*/

void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/* USER CODE BEGIN 0 */

/* ========================================================================== */
/* Utility functions                                                         */
/* ========================================================================== */

static uint16_t crc16_ccitt(const uint8_t *data, uint32_t len)
{
    uint16_t crc = 0xFFFFU;

    for (uint32_t i = 0U; i < len; i++)
    {
        crc ^= (uint16_t)data[i] << 8;

        for (uint8_t bit = 0U; bit < 8U; bit++)
        {
            if (crc & 0x8000U)
            {
                crc = (uint16_t)((crc << 1) ^ 0x1021U);
            }
            else
            {
                crc <<= 1;
            }
        }
    }

    return crc;
}

static void write_u16_le(uint8_t **p, uint16_t value)
{
    (*p)[0] = (uint8_t)(value & 0xFFU);
    (*p)[1] = (uint8_t)(value >> 8);
    *p += 2;
}

static void write_i16_le(uint8_t **p, int16_t value)
{
    write_u16_le(p, (uint16_t)value);
}

static uint16_t clamp_u16(float value)
{
    if (value <= 0.0f)
        return 0U;

    if (value >= 65535.0f)
        return 65535U;

    return (uint16_t)value;
}

static uint8_t is_valid_stream_mode(uint8_t mode)
{
    return (mode == STREAM_MODE_FFT ||
            mode == STREAM_MODE_RAW ||
            mode == STREAM_MODE_SENSORS ||
            mode == STREAM_MODE_OSC);
}

static void write_frame_header(uint8_t **p, uint16_t seq)
{
    *(*p)++ = PROTO_SYNC0;
    *(*p)++ = PROTO_SYNC1;

    write_u16_le(p, seq);

    *(*p)++ = NUM_CHANNELS;

    write_u16_le(p, DISPLAY_BINS);
}

static uint16_t finish_frame(uint8_t *frame_buf, uint8_t *p)
{
    uint16_t crc = crc16_ccitt(
        frame_buf + PROTO_HDR_SIZE,
        PROTO_DATA_BYTES
    );

    write_u16_le(&p, crc);

    return (uint16_t)(p - frame_buf);
}

/* ========================================================================== */
/* Sample timing                                                             */
/* ========================================================================== */

/*
 * 25 frames = 500 ms.
 *
 * 128 fresh samples are distributed as:
 *
 *     3 x 6 samples
 *    22 x 5 samples
 *
 * Frames 0, 12 and 24 contain 6 fresh samples.
 */
static uint16_t fresh_samples_for_frame(uint16_t seq)
{
    uint16_t slot = seq % 25U;

    if (slot == 0U || slot == 12U || slot == 24U)
        return 6U;

    return 5U;
}

/* ========================================================================== */
/* Test waveforms                                                             */
/* ========================================================================== */

static float eeg_test_sample(uint16_t ch, uint32_t sample_index)
{
    float t = (float)sample_index / SAMPLE_RATE;
    float phase = (float)ch * 0.17f;

    /* ADS1299-style placeholder: centered EEG in microvolts. */
    return
        28.0f * sinf(2.0f * (float)M_PI * 10.0f * t + phase) +
        10.0f * sinf(2.0f * (float)M_PI * 2.0f * t + phase * 0.5f) +
         7.0f * sinf(2.0f * (float)M_PI * 20.0f * t + phase * 1.7f) +
         2.0f * sinf(2.0f * (float)M_PI * 37.0f * t + phase * 2.3f);
}

static float sensors_test_sample(uint16_t ch)
{
    return 10000.0f + (float)ch * 1500.0f;
}

static float osc_test_sample(uint16_t ch, uint32_t sample_index)
{
    return 120.0f * sinf(
        2.0f * (float)M_PI *
        ((float)sample_index * 25.0f / SAMPLE_RATE + (float)ch * 0.15f)
    );
}

/* ========================================================================== */
/* Frame builders                                                             */
/* ========================================================================== */

static uint16_t build_raw_frame(uint8_t *frame_buf)
{
    uint8_t *p = frame_buf;

    uint16_t seq = g_seq++;
    uint32_t start = g_raw_sample_index;

    write_frame_header(&p, seq);

    for (uint16_t ch = 0U; ch < NUM_CHANNELS; ch++)
    {
        for (uint16_t n = 0U; n < DISPLAY_BINS; n++)
        {
            int16_t value =
                (int16_t)lroundf(eeg_test_sample(ch, start + n));

            write_i16_le(&p, value);
        }
    }

    g_raw_sample_index += fresh_samples_for_frame(seq);

    return finish_frame(frame_buf, p);
}

static uint16_t build_fft_frame(uint8_t *frame_buf)
{
    uint8_t *p = frame_buf;

    uint16_t seq = g_seq++;
    uint32_t start = g_raw_sample_index;

    write_frame_header(&p, seq);

    for (uint16_t ch = 0U; ch < NUM_CHANNELS; ch++)
    {
        for (uint16_t n = 0U; n < FFT_LENGTH; n++)
        {
            fft_input[n] = eeg_test_sample(ch, start + n);
        }

        arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);

        for (uint16_t bin = 0U; bin < DISPLAY_BINS; bin++)
        {
            float magnitude = bin == 0U
                ? fabsf(fft_output[0])
                : sqrtf(
                    fft_output[2U * bin] * fft_output[2U * bin] +
                    fft_output[2U * bin + 1U] * fft_output[2U * bin + 1U]
                );
            uint16_t value = clamp_u16(2.0f * magnitude / FFT_LENGTH);

            write_u16_le(&p, value);
        }
    }

    g_raw_sample_index += fresh_samples_for_frame(seq);
    return finish_frame(frame_buf, p);
}

static uint16_t build_sensors_frame(uint8_t *frame_buf)
{
    uint8_t *p = frame_buf;

    uint16_t seq = g_seq++;

    write_frame_header(&p, seq);

    for (uint16_t ch = 0U; ch < NUM_CHANNELS; ch++)
    {
        uint16_t value = clamp_u16(sensors_test_sample(ch));

        for (uint16_t n = 0U; n < DISPLAY_BINS; n++)
        {
            write_u16_le(&p, value);
        }
    }

    g_raw_sample_index += fresh_samples_for_frame(seq);

    return finish_frame(frame_buf, p);
}

static uint16_t build_osc_frame(uint8_t *frame_buf)
{
    uint8_t *p = frame_buf;

    uint16_t seq = g_seq++;
    uint32_t start = g_raw_sample_index;

    write_frame_header(&p, seq);

    for (uint16_t ch = 0U; ch < NUM_CHANNELS; ch++)
    {
        for (uint16_t n = 0U; n < DISPLAY_BINS; n++)
        {
            int16_t value =
                (int16_t)lroundf(osc_test_sample(ch, start + n));

            write_i16_le(&p, value);
        }
    }

    g_raw_sample_index += fresh_samples_for_frame(seq);

    return finish_frame(frame_buf, p);
}

/* ========================================================================== */
/* Stream frame selector                                                      */
/* ========================================================================== */

static uint16_t build_stream_frame(
    uint8_t mode,
    uint8_t *frame_buf)
{
    switch (mode)
    {
        case STREAM_MODE_RAW:
            return build_raw_frame(frame_buf);

        case STREAM_MODE_FFT:
            return build_fft_frame(frame_buf);

        case STREAM_MODE_SENSORS:
            return build_sensors_frame(frame_buf);

        case STREAM_MODE_OSC:
            return build_osc_frame(frame_buf);

        default:
            return 0U;
    }
}

/* USER CODE END 0 */


/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
    HAL_Init();

    SystemClock_Config();
    MX_GPIO_Init();
    MX_USB_DEVICE_Init();
    arm_rfft_fast_init_f32(&fft_instance, FFT_LENGTH);

    uint8_t frame_index = 0U;
    uint8_t frame_ready = 0U;

    uint8_t stream_mode = STREAM_MODE_FFT;

    uint16_t frame_len = 0U;

    uint32_t next_build =
        HAL_GetTick() + FRAME_PERIOD_MS;

    while (1)
    {
        uint32_t now = HAL_GetTick();

        /* ================================================================ */
        /* Check USB mode command                                           */
        /* ================================================================ */

        uint8_t requested_mode = CDC_GetStreamMode();

        if (is_valid_stream_mode(requested_mode) &&
            requested_mode != stream_mode)
        {
            /*
             * Mode changed.
             */
            stream_mode = requested_mode;

            /*
             * Restart stream counters.
             */
            g_seq = 0U;
            g_raw_sample_index = 0U;

            /*
             * Throw away any frame belonging to the previous mode.
             */
            frame_ready = 0U;

            /*
             * Start the new stream on the next 20 ms boundary.
             */
            next_build = now;
        }

        /* ================================================================ */
        /* Build frame                                                      */
        /* ================================================================ */

        if (!frame_ready &&
            (int32_t)(now - next_build) >= 0)
        {
            frame_len =
                build_stream_frame(
                    stream_mode,
                    frame[frame_index]
                );

            /*
             * If an invalid mode somehow reaches here,
             * don't transmit an empty frame.
             */
            if (frame_len != 0U)
            {
                frame_ready = 1U;
            }

            next_build += FRAME_PERIOD_MS;

            /*
             * Avoid building a burst of old frames
             * if the CPU was delayed.
             */
            if ((int32_t)(now - next_build) >
                (int32_t)FRAME_PERIOD_MS)
            {
                next_build = now + FRAME_PERIOD_MS;
            }
        }

        /* ================================================================ */
        /* USB transmit                                                      */
        /* ================================================================ */

        if (frame_ready)
        {
            if (CDC_Transmit_FS(
                    frame[frame_index],
                    frame_len) == USBD_OK)
            {
                /*
                 * Only change buffer after USB accepted
                 * the current frame.
                 */
                frame_index ^= 1U;
                frame_ready = 0U;
            }
        }
    }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 72;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_2, GPIO_PIN_RESET);

  /*Configure GPIO pin : PB2 */
  GPIO_InitStruct.Pin = GPIO_PIN_2;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */