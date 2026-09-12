/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"
#include "usbd_cdc_if.h"
#include <math.h>
#include <string.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define PROTO_SYNC0      0xAAU
#define PROTO_SYNC1      0x55U
#define NUM_CHANNELS     32U
#define DISPLAY_BINS     128U
#define PROTO_HDR_SIZE   7U
#define FRAME_PERIOD_MS  20U
#define SAMPLE_RATE      256U
#define PROTO_DATA_BYTES (NUM_CHANNELS * DISPLAY_BINS * 2U)
#define PROTO_FRAME_SIZE (PROTO_HDR_SIZE + PROTO_DATA_BYTES + 2U)
#define STREAM_MODE_FFT  'F'
#define STREAM_MODE_RAW  'R'
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
static uint16_t g_seq = 0U;
static uint32_t g_raw_sample_index = 0U;

/* Enough room for the largest supported mode (FFT). */
static uint8_t frame[2][PROTO_FRAME_SIZE];

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static uint16_t crc16_ccitt(const uint8_t *data, uint32_t len)
{
  uint16_t crc = 0xFFFFU;
  for (uint32_t i = 0; i < len; ++i)
  {
    crc ^= (uint16_t)data[i] << 8;
    for (uint8_t bit = 0; bit < 8U; ++bit)
    {
      crc = (crc & 0x8000U) ? (uint16_t)((crc << 1) ^ 0x1021U)
                             : (uint16_t)(crc << 1);
    }
  }
  return crc;
}

static void write_u16_le(uint8_t **pp, uint16_t value)
{
  uint8_t *p = *pp;
  p[0] = (uint8_t)(value & 0xFFU);
  p[1] = (uint8_t)(value >> 8);
  *pp += 2;
}

static uint16_t clamp_u16(float value)
{
  if (value <= 0.0f) return 0U;
  if (value >= 65535.0f) return 65535U;
  return (uint16_t)value;
}

/* Test waveform only. Replace this with your ADC/DMA samples in the real build. */
static float raw_test_sample(uint16_t ch, uint32_t sample_index)
{
  const float phase = (float)ch * 0.20f;
  const float signal = sinf(2.0f * (float)M_PI *
                            ((float)sample_index * 8.0f / (float)SAMPLE_RATE + phase));
  return 32768.0f + signal * 24000.0f;
}

/* Test FFT data only. */
static float fft_test_sample(uint16_t ch, uint16_t bin, float t)
{
  const float freq_hz = (float)bin;
  const float offset = (float)ch * 0.6f;
  const float targets[4] = {8.0f + offset, 18.0f + offset,
                            32.0f + offset, 58.0f + offset};
  const float widths[4] = {1.3f, 1.4f, 2.0f, 2.5f};
  float total = 0.0f;

  for (uint8_t i = 0; i < 4U; ++i)
  {
    const float delta = freq_hz - targets[i];
    total += 28000.0f * expf(-(delta * delta) /
                             (2.0f * widths[i] * widths[i]));
  }

  const float envelope = 0.8f +
      0.2f * sinf(2.0f * (float)M_PI *
                  (t * 0.5f + (float)ch * 0.13f));
  return envelope * total + 200.0f;
}

/*
 * RAW packet:
 *   Uses the same 32 x 128 payload shape as FFT packets so the existing
 *   Python parser can consume both modes without changing frame boundaries.
 * The waveform clock advances by the complete 128-sample block because the
 * current Python protocol appends every value in the fixed-size packet.
 */
static uint16_t build_raw_frame(uint8_t *frame_buf)
{
  uint8_t *p = frame_buf;
  const uint16_t seq = g_seq++;
  const uint32_t start = g_raw_sample_index;

  *p++ = PROTO_SYNC0;
  *p++ = PROTO_SYNC1;
  write_u16_le(&p, seq);
  *p++ = (uint8_t)NUM_CHANNELS;
  write_u16_le(&p, DISPLAY_BINS);

  for (uint16_t ch = 0; ch < NUM_CHANNELS; ++ch)
  {
    for (uint16_t n = 0; n < DISPLAY_BINS; ++n)
    {
      write_u16_le(&p, clamp_u16(raw_test_sample(ch, start + n)));
    }
  }

  write_u16_le(&p, crc16_ccitt(frame_buf + PROTO_HDR_SIZE, PROTO_DATA_BYTES));
  g_raw_sample_index += DISPLAY_BINS;
  return (uint16_t)(p - frame_buf);
}

static uint16_t build_fft_frame(uint8_t *frame_buf)
{
  uint8_t *p = frame_buf;
  const uint16_t seq = g_seq++;
  const float t = (float)HAL_GetTick() / 1000.0f;

  *p++ = PROTO_SYNC0;
  *p++ = PROTO_SYNC1;
  write_u16_le(&p, seq);
  *p++ = (uint8_t)NUM_CHANNELS;
  write_u16_le(&p, DISPLAY_BINS);

  for (uint16_t ch = 0; ch < NUM_CHANNELS; ++ch)
  {
    for (uint16_t bin = 0; bin < DISPLAY_BINS; ++bin)
    {
      write_u16_le(&p, clamp_u16(fft_test_sample(ch, bin, t)));
    }
  }

  write_u16_le(&p, crc16_ccitt(frame_buf + PROTO_HDR_SIZE, PROTO_DATA_BYTES));
  return (uint16_t)(p - frame_buf);
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

  uint8_t frame_index = 0U;
  uint8_t frame_ready = 0U;
  uint8_t stream_mode = STREAM_MODE_RAW;
  uint16_t frame_len = 0U;
  uint32_t next_build = HAL_GetTick() + FRAME_PERIOD_MS;

  while (1)
  {
    const uint32_t now = HAL_GetTick();

    const uint8_t requested_mode = CDC_GetStreamMode();
    if (requested_mode == STREAM_MODE_RAW || requested_mode == STREAM_MODE_FFT)
    {
      if (requested_mode != stream_mode)
      {
        stream_mode = requested_mode;
        g_raw_sample_index = 0U;
        next_build = now + FRAME_PERIOD_MS;
      }
    }

    /* Build once. If USB is busy, this exact frame is retried; it is never rebuilt. */
    if (!frame_ready && (int32_t)(now - next_build) >= 0)
    {
      if (stream_mode == STREAM_MODE_RAW)
      {
        frame_len = build_raw_frame(frame[frame_index]);
      }
      else
      {
        frame_len = build_fft_frame(frame[frame_index]);
      }

      frame_ready = 1U;
      next_build += FRAME_PERIOD_MS;

      /* If the CPU was delayed for a long time, don't burst-build old frames. */
      if ((int32_t)(now - next_build) > (int32_t)FRAME_PERIOD_MS)
      {
        next_build = now + FRAME_PERIOD_MS;
      }
    }

    /* USB CDC is asynchronous. Only advance the double buffer after success. */
    if (frame_ready && CDC_Transmit_FS(frame[frame_index], frame_len) == USBD_OK)
    {
      frame_index ^= 1U;
      frame_ready = 0U;
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